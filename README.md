# Network Anomaly Detection With Temporal Convolutional Network and U-Net Model
This is code reproduced according to the description in the paper "[Network Anomaly Detection With Temporal Convolutional Network and U-Net Model](https://arxiv.org/pdf/1805.03409.pdf)".    

Using 1D TCN and 1D U-Net.
## Dependencies
### Environment
- OS: Ubuntu 22.04.2 LTS
- CPU: AMD EPYC 7453 28-Core Processor, CPU(s): 112
- GPU: NVIDIA A100 PCIe 80GB / NVIDIA A100 PCIe 80GB
- Memory: 256GB

### Packages
- python 3.9.16
- numpy 1.23.4
- pandas 1.3.3
- keras 2.6.0
- scikit-learn 1.3.0
- tensorflow 2.6.0
- tensorflow-gpu 2.6.0

Run the following command to install the required packages (Please make sure your CUDA version is compatible with the tensorflow version) :

```
pip install -r requirements.txt
```
## Usage guide
1. First, please download the dataset from [CSE-CIC-IDS2018 on AWS](https://www.unb.ca/cic/datasets/ids-2018.html), or you can use the data in the `sample_data` folder for program testing.

2. Then you can directly use the python or python3 command to execute the program, for example: 

```
python3 U-Net.py
```

3. I wrote the training and testing together. Once the program is completed, each Average (`marco`, `mirco`, `weighted`) will be obtained, and the Score Report of each category will also be obtained. In addition, although the paper does not mention which indicator is used to calculate the Average, based on the experimental results, I guess that `Weighted` is used to calculate the Average.

## Reference
Mezina A, Burget R, Travieso-González CM. Network Anomaly Detection With Temporal Convolutional Network and U-Net Model. In: IEEE Access (Volume:9).