# Extreme Classification [![Build Status](https://travis-ci.org/vishwakftw/extreme-classification.svg?branch=master)](https://travis-ci.org/vishwakftw/extreme-classification)

`extreme_classification` is a Python module designed for extreme classification tasks with two new algorithms:
- **NeuralXC**: A deep-learning based solution using autoencoders and neural networks.

- **HierarchicalXC**: A hierarchical clustering based approach

## Using the module

### Setup

- Requirements for the project are listed in `requirements.txt`. In addition to these, it is mandatory to have PyTorch installed in your system with version 0.4.1 or higher.
   - Requirements can be installed using `pip`:
     ```bash
     $ pip install -r requirements.txt 
     ```
     or using `conda`:
     ```bash
     $ conda install --file requirements.txt
     ```

### Installing `extreme_classification`

- Clone the repository.

- Run `python setup.py install`.

- To test if your installation is successful, try running the command:
```bash
$ python -c "import extreme_classification"
```

## Using the training scripts

- `train_neuralxc.py` is a Python script used to train an extreme classification model using **NeuralXC**. You can run the script by passing required options:
```bash
python train_neuralxc.py [-h] --data_root DATA_ROOT --dataset_info DATASET_INFO
                         --input_encoder_cfg INPUT_ENCODER_CFG --input_decoder_cfg INPUT_DECODER_CFG
                         --output_encoder_cfg OUTPUT_ENCODER_CFG --output_decoder_cfg OUTPUT_DECODER_CFG
                         --regressor_cfg REGRESSOR_CFG --optimizer_cfg OPTIMIZER_CFG
                         [--init_scheme {xavier_uniform,kaiming_uniform,default}]
                         [--device DEVICE] [--seed SEED] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                         [--interval INTERVAL] [--input_ae_loss_weight INPUT_AE_LOSS_WEIGHT]
                         [--output_ae_loss_weight OUTPUT_AE_LOSS_WEIGHT] [--k K]
                         [--plot] [--save_model {all,inputAE,outputAE,regressor} [{all,inputAE,outputAE,regressor} ...]]
```
For more information about the options, please run `python train_neuralxc.py -h`.

- `train_hierarchicalXC.py` is a Python script to train an extreme classification model using **HierarchicalXC**. You can run the script by passing required options:
```bash
python train_hierarchicalXC.py [-h] --data_root DATA_ROOT --dataset_info
                               DATASET_INFO [--device DEVICE] [--seed SEED]
                               [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                               [--interval INTERVAL] [--input_ae_dim INPUT_AE_DIM]
                               [--njobs NJOBS] [--optimizer_cfg OPTIMIZER_CFG]
                               [--save_model] [--k K]
```
For more information about the options, please run `python train_hierarchicalxc.py -h`.


### Configuration files

#### Neural Network Configurations
- For `train_neuralxc.py`, you have to have valid neural network configurations for the autoencoders of the inputs and outputs and the regressor in YAML. An example of a configuration file would be this:
```yaml
- name: Linear
  kwargs:
    in_features: 500
    out_features: 1152

- name: LeakyReLU
  kwargs:
    negative_slope: 0.2
    inplace: True

- name: Linear
  kwargs:
    in_features: 1152
    out_features: 1836

- name: Sigmoid
```
Please note that the `name` and `kwargs` attributes have to resemble the same names as those in PyTorch.

#### Optimizer Configurations
- Optimizer configurations are very similar to the neural network configurations. Here you have to retain the same naming as PyTorch for optimizer names and their parameters - for example: `lr` for learning rate. Below is a sample:
```yaml
name: Adam
args:
  lr: 0.001
  betas: [0.5, 0.9]
  weight_decay: 0.0001
```

#### Dataset Configurations
- In both the scripts, you are required to specify a data root (`data_root`), dataset information file (`dataset_info`). `data_root` corresponds to the folder containing the datasets. `dataset_info` requires a YAML file in the following format:
```yaml
train_filename:
train_opts:
  num_data_points:
  input_dims:
  output_dims:

test_filename:
test_opts:
  num_data_points:
  input_dims:
  output_dims:
```

- If the test dataset doesn't exist, then please remove the fields `test_filename` and `test_opts`. An example for the Bibtex dataset would be:
```yaml
train_filename: bibtex_train.txt
train_opts:
  num_data_points: 4880
  input_dims: 1836
  output_dims: 159

test_filename: bibtex_test.txt
test_opts:
  num_data_points: 2515
  input_dims: 1836
  output_dims: 159
```

##### This is a project designed for CS6370 : Information Retrieval offered in Fall 2018
