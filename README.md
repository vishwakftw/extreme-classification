# Extreme Classification [![Build Status](https://travis-ci.org/vishwakftw/extreme-classification.svg?branch=master)](https://travis-ci.org/vishwakftw/extreme-classification)

`extreme_classification` is a Python module designed for extreme classification tasks with two new algorithms:
- **NeuralXC**: A deep-learning based solution using autoencoders and neural networks.

- **HierarchicalXC**: A hierarchical clustering based approach

This project also includes scripts for training and testing on datasets using this module.

## Setup

### Prerequisites

- Python 2.7 or 3.5
- Requirements for the project are listed in [requirements.txt](requirements.txt). In addition to these, PyTorch 0.4.1 or higher is necessary. The requirements can be installed using pip:
   ```bash
   $ pip install -r requirements.txt 
   ```
   or using conda:
   ```bash
   $ conda install --file requirements.txt
     ```

### Installing `extreme_classification`

- Clone the repository.
  ```bash
  $ git clone https://github.com/vishwakftw/extreme-classification
  $ cd extreme-classification
  ```

- Install
  ```bash
  $ python setup.py install
  ```

- To test if your installation is successful, try running the command:
  ```bash
  $ python -c "import extreme_classification"
  ```

## Using the scripts

### NeuralXC

Use `train_neuralxc.py`. A description of the options available can be found using:

```bash
python train_neuralxc.py --help
```

This script trains (and optionally evaluates) evaluates a model on a given dataset using the NeuralXC algorithm.

### HierarchicalXC

Use `train_hierarchicalXC.py`. A description of the options available can be found using:
```bash
python train_hierarchicalXC.py --help
```
This script trains (and optionally evaluates) evaluates a model on a given dataset using the HierarchicalXC algorithm.

## Data Format
The input data must be in the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format. An example of such a dataset is the Bibtex dataset found [here](http://manikvarma.org/downloads/XC/XMLRepository.html).

The first row in the LIBSVM format specifies dataset files and input and output dimensions. This row must be removed, and this information must be provided through configuration files, as explained below.

## Configuration files

### Neural Network Configurations
For using NeuralXC through `train_neuralxc.py`, you need to have valid neural network configurations for the autoencoders of the inputs, labels and the regressor in the YAML format. An example configuration file is:
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

### Optimizer Configurations
Optimizer configurations are very similar to the neural network configurations. Here you have to retain the same naming as PyTorch for optimizer names and their parameters - for example: `lr` for learning rate. Below is a sample:
```yaml
name: Adam
args:
  lr: 0.001
  betas: [0.5, 0.9]
  weight_decay: 0.0001
```

### Dataset Configurations
In both the scripts, you are required to specify a data root (`data_root`), dataset information file (`dataset_info`). `data_root` corresponds to the folder containing the datasets. `dataset_info` requires a YAML file in the following format:
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

If the test dataset doesn't exist, then please remove the fields `test_filename` and `test_opts`. An example for the Bibtex dataset would be:
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

## License
This code is provided under the [MIT License](LICENSE)

---
This project was a part of the course CS6370: Information Retrieval offered in Fall 2018 at IIT Hyderabad
