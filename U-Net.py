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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def GPU_clean():
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def train():
    GPU_clean()
    print("=============== Data loading 🚗... ===============")
    df = pd.concat((pd.read_csv(f) for f in iglob('sample_data/*.csv', recursive=True)), ignore_index=True)
    print("=============== Data loading has completed ✅ ===============")
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

        model = unet_model(input_dim)
        opt = Adam(learning_rate=0.0001)
        model.compile(loss = 'categorical_crossentropy',
                    optimizer = opt, 
                    metrics=['accuracy'])

        cp = ModelCheckpoint(filepath=f"models/U-Net.h5",
                                save_best_only=True,
                                verbose=0)
        
        tb = TensorBoard(log_dir=f"./logs",
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)
        
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


def unet_model(input_dim):
    print("input_dim: ", input_dim)
    inputs = tf.keras.Input(shape=(input_dim, 1))
    zeropadding = layers.ZeroPadding1D(padding=1)(inputs)

    ''' ========== Encoder ========== '''
    # Block 1
    conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')(zeropadding)
    conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2, strides=2)(conv1)
    # Block 2
    conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling1D(pool_size=2, strides=2)(conv2)
    # Block 3
    conv3 = layers.Conv1D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv1D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling1D(pool_size=2, strides=2)(conv3)
    # Block 4
    conv4 = layers.Conv1D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv1D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling1D(pool_size=2, strides=2)(conv4)
    # Block 5
    conv5 = layers.Conv1D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv1D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = layers.Dropout(0.5)(conv5)

    ''' ========== Decoder ========== '''
    # Block 6
    up6 = layers.UpSampling1D(size=2)(conv5)
    up6_half = int(up6.shape[1] / 2) 
    crop_up6 = layers.Cropping1D(cropping=(up6_half, 0))(up6) 
    concat6 = layers.Concatenate()([pool4, crop_up6]) 
    conv6 = layers.Conv1D(512, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv1D(512, 3, activation='relu', padding='same')(conv6)
    # Block 7
    up7 = layers.UpSampling1D(size=2)(conv6)
    concat7 = layers.Concatenate()([pool3, up7]) 
    conv7 = layers.Conv1D(256, 3, activation='relu', padding='same')(concat7)
    conv7 = layers.Conv1D(256, 3, activation='relu', padding='same')(conv7)
    # Block 8
    up8 = layers.UpSampling1D(size=2)(conv7)
    concat8 = layers.Concatenate()([pool2, up8]) 
    conv8 = layers.Conv1D(128, 3, activation='relu', padding='same')(concat8)
    conv8 = layers.Conv1D(128, 3, activation='relu', padding='same')(conv8)
    # Block 9
    up9 = layers.UpSampling1D(size=2)(conv8)
    concat9 = layers.Concatenate()([pool1, up9]) 
    conv9 = layers.Conv1D(64, 3, activation='relu', padding='same')(concat9)
    conv9 = layers.Conv1D(64, 3, activation='relu', padding='same')(conv9)

    # Flatten
    flatten = layers.Flatten()(conv9)

    # Dense layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(32, activation='relu')(dense1)

    # Output layer
    output = layers.Dense(labels_len, activation='softmax')(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    # model.summary()

    return model

if __name__ == "__main__":
    train()
