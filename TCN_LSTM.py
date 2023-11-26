import os, time, copy
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import iglob
from tensorflow.keras.optimizers import Adam
from keras import layers
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from losses import categorical_focal_loss
from channel_normalize import channel_normalization

"""

The following section of code is referenced from:

    [losses.py](https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py) by [Umberto Griffo]
    [channel_normalize.py](https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/tf_models.py) by [Colin Lea]

"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def GPU_clean():
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def train():
    GPU_clean()
    print("=============== Data loading 🚗... ===============")
    df = pd.concat((pd.read_csv(f) for f in iglob('sample_data/*.csv', recursive=True)), ignore_index=True)
    print("=============== Data loading is complete ✅ ===============")
    # print(df['Label'].unique())
    
    print("=============== Remove redundant features 🚗... ===============")
    columns_to_drop = ['Unnamed: 0', 'Timestamp', 'Flow ID']
    for col in columns_to_drop:
        try:
            df = df.drop(col, axis='columns')
            print(f"Column '{col}' has been dropped.")
        except KeyError:
            print(f"Column '{col}' does not exist.")

    for f in df.columns: 
        if df[f].dtype=='object': 
            label = LabelEncoder() 
            label.fit(list(df[f].values)) 
            df[f] = label.transform(list(df[f].values))

    df = df.fillna(value=0)
    df = df.replace([np.inf, -np.inf], 0)
    df_nolabel = df.drop(columns=['Label'])

    global labels_len
    x1 = df_nolabel
    y1 = df['Label'] # Label
    labels_len = len(np.unique(y1))
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.1, random_state=42)
    print("=============== Data split OK ✅ ===============")

    # k-fold
    kflod = KFold(n_splits=5, shuffle=True, random_state=42)
    
    training_time_total = []
    testing_time_predict = []
    accuracy_scores = []
    # roc_auc_scores = []
    recall_scores_weighted = []
    precision_scores_weighted = []
    f1_scores_weighted = []
    recall_scores_micro = []
    precision_scores_micro = []
    f1_scores_micro = []
    recall_scores_macro = []
    precision_scores_macro = []
    f1_scores_macro = []

    temp_x = x_test.copy() # Ensure that each loop is processed using the original data.
    temp_y = y_test.copy()
    x_train_k = x_train.copy() # Ensure that each loop is processed using the original data.
    y_train_k = y_train.copy()

    for train_index, val_index in kflod.split(x_train_k, y_train_k):
        x_test = temp_x
        y_test = temp_y

        x_train, x_val = x_train_k.iloc[train_index], x_train_k.iloc[val_index]
        y_train, y_val = y_train_k.iloc[train_index], y_train_k.iloc[val_index]

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_train = tf.expand_dims(x_train, axis=2)

        x_val = scaler.transform(x_val)
        x_val = tf.expand_dims(x_val, axis=2)
        y_val = tf.one_hot(y_val, depth=labels_len) # Set class number.

        x_test = scaler.transform(x_test)
        x_test = tf.expand_dims(x_test, axis=2)
        y_test = tf.one_hot(y_test, depth=labels_len) # Set class number.

        y_train = y_train.to_numpy()
        y_train = tf.one_hot(y_train, depth=labels_len) # Set class number.

        input_dim = x_train.shape[1] # Get input shape.
        num_epochs = 10
        batch_size = 512

        model = TCN_LSTM(input_dim)
        opt = Adam(learning_rate=0.00001)
        loss = [categorical_focal_loss(alpha=0.25, gamma=2)]
        model.compile(loss = loss,
                    optimizer = opt, 
                    metrics=['accuracy'])

        cp = ModelCheckpoint(filepath=f"models/TCN_LSTM.h5",
                                save_best_only=True,
                                verbose=0)
        
        tb = TensorBoard(log_dir=f"./logs",
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True) # Save a logs.
        
        start = time.time()
        model.fit(x_train, y_train, 
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                verbose=1,
                callbacks=[cp, tb])
        end = time.time()
        training_time_total.append(end - start)

        testing_start_predict = time.time()
        y_pred = model.predict(x_test)
        testing_end_predict = time.time()
        testing_time_predict.append(testing_end_predict - testing_start_predict)
        
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # y_true_prob = tf.one_hot(y_true, depth=labels_len) # Used to calculate ROC_AUC score
        # y_pred_prob = tf.one_hot(y_pred, depth=labels_len) # Used to calculate ROC_AUC score

        print("=============== Calculating Score 🚗... ===============")
        report = classification_report(y_true, y_pred)
        accuracy_scores.append(accuracy_score(y_true, y_pred))
        # roc_auc_scores.append(roc_auc_score(y_true_prob, y_pred_prob, average='weighted', multi_class='ovr'))
        precision_scores_weighted.append(precision_score(y_true, y_pred, average='weighted'))
        recall_scores_weighted.append(recall_score(y_true, y_pred, average='weighted'))
        f1_scores_weighted.append(f1_score(y_true, y_pred, average='weighted'))

        precision_scores_macro.append(precision_score(y_true, y_pred, average='macro'))
        recall_scores_macro.append(recall_score(y_true, y_pred, average='macro'))
        f1_scores_macro.append(f1_score(y_true, y_pred, average='macro'))

        precision_scores_micro.append(precision_score(y_true, y_pred, average='micro'))
        recall_scores_micro.append(recall_score(y_true, y_pred, average='micro'))
        f1_scores_micro.append(f1_score(y_true, y_pred, average='micro'))
    
    print("=============== Score calculation has completed ✅ ===============")

   # print("ROC_AUC_SCORE: %f" % np.mean(roc_auc_scores))
    print("Report:\n", report)
    print("Accuracy average: %f" % np.mean(accuracy_scores))
    print("=========================================")
    print("Precision average_weighted: %f" % np.mean(precision_scores_weighted))
    print("Recall average_weighted: %f" % np.mean(recall_scores_weighted))
    print("F1_score average_weighted: %f" % np.mean(f1_scores_weighted))
    print("=========================================")
    print("Precision average_macro: %f" % np.mean(precision_scores_macro))
    print("Recall average_macro: %f" % np.mean(recall_scores_macro))
    print("F1_score average_macro: %f" % np.mean(f1_scores_macro))
    print("=========================================")
    print("Precision average_micro: %f" % np.mean(precision_scores_micro))
    print("Recall average_micro: %f" % np.mean(recall_scores_micro))
    print("F1_score average_micro: %f" % np.mean(f1_scores_micro))
    print("=========================================")
    print("Average training time: %f Sec" % np.mean(training_time_total))
    print("Average predict time: %f Sec" % np.mean(testing_time_predict))

def TCN_LSTM(input_dim):
    print("input_dim: ", input_dim)

    ''' ========== TCN Encoder ========== '''
    input = tf.keras.Input(shape=(input_dim, 1))

    # Block 1
    x = layers.Conv1D(filters=64, kernel_size=4, dilation_rate=1, padding='same')(input)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Activation('relu')(x)
    encoder_output = layers.Lambda(channel_normalization)(x)

    # Block 2
    x = layers.Conv1D(filters=96, kernel_size=4, dilation_rate=2, padding='same')(encoder_output)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Activation('relu')(x)
    encoder_output = layers.Lambda(channel_normalization)(x)

    ''' ========== TCN Decoder ========== '''
    # Block 3
    x = layers.Conv1D(filters=96, kernel_size=4, dilation_rate=2, padding='same')(encoder_output)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Activation('relu')(x)
    encoder_output = layers.Lambda(channel_normalization)(x)

    # Block 4
    x = layers.Conv1D(filters=64, kernel_size=4, dilation_rate=1, padding='same')(encoder_output)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Activation('relu')(x)
    encoder_output = layers.Lambda(channel_normalization)(x)

    # FCN
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(encoder_output) # 443040, 779,552, 426,656
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(64, activation='LeakyReLU')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, activation='LeakyReLU')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(16, activation='LeakyReLU')(x)
    x = layers.Dropout(0.5)(x)

    fcn_output = layers.Dense(labels_len, activation='softmax')(x) 

    # LSTM
    y = layers.LSTM(units=8)(input)
    lstm_output = layers.Dropout(0.5)(y)

    # Concatenate TCN and LSTM outputs
    combined = layers.Concatenate()([fcn_output, lstm_output])

    output = layers.Dense(labels_len, activation='softmax')(combined)
    model = tf.keras.Model(inputs=input, outputs=output)
    # model.summary()

    return model

if __name__ == "__main__":
    train()

