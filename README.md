# Various Algorithms & Software Tools (VAST)
This repository contains some common functionalities used in various works from the members of the
Vision And Security Technology (VAST) Lab.

## Setup
Clone with sub modules such as `FINCH`
```git clone --recurse-submodules https://github.com/Vastlab/vast.git```

Since this repository is being used as a package in various other repositories, it is recommended to
export the path this repository is cloned into the `PYTHONPATH` variable as:
```export PYTHONPATH="{PARENT_DIR_WHERE_REPO_IS_CLONED}:"```
To make the change permanent please add the above to your `.bashrc`

## Contents:
### Loss Functions
1. Entropic Openset loss
2. Objectosphere loss
3. Center loss
4. Objecto-center loss (Objectosphere + Center loss)

### Network Architectures
1. LeNet
2. LeNet++

### Openset Algorithms
1. OpenMax
2. Multimodal OpenMax
3. Extreme Value Machine (EVM)

### Reimplementation of libMR
This repo contains a torch based reimplementation of the [libMR repo](https://github.com/Vastlab/libMR)
It supports GPU based computation that speeds up the processing considerably, but in certain cases the weibull parameter
computation may have slight variations.

### Tools
1. [Concatenate multiple torch datasets](https://github.com/Vastlab/vast/blob/main/tools/ConcatDataset.py) Useful for openset learning.
2. [Feature Extraction](https://github.com/Vastlab/vast/blob/main/tools/FeatureExtraction.py) to HDF5 file from a specific layer of a pytorch model
3. Multiprocessing Logger

### Visualization
1. 2D visualization e.g. features from LeNet++
2. OSRC curve plotter
3. Histogram of scores

### Evaluation
1. OSRC curve using torch tensors and cuda operations
2. FPR vs coverage plot
3. Precision/Recall for binary class OOD problem
4. F-ß Score




## Examples
Some research works using this repo are:
[Self-Supervised Features Improve Open-World Learning](https://github.com/Vastlab/SSFiOWL)
[MNIST Based Experiments](https://github.com/Vastlab/MNIST_Experiments)
[ImageNet Level Openset experiments](https://github.com/Vastlab/ImageNetDali)
