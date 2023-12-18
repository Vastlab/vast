# Various Algorithms & Software Tools (VAST)
This repository contains some common functionalities used in various works from the members of the
Vision And Security Technology (VAST) Lab.

## Setup
### For users
`pip install git+https://github.com/Vastlab/vast.git`

**NOTE:** There is an unadressed issue due to which the above install makes features like FINCH unable to end user.
If you intend to use FINCH please follow the for developers instructions below.
### For developers
`git clone --recurse-submodules https://github.com/Vastlab/vast.git`

`pip install -e .[dev]`

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
[Reimplementation of libMR](vast/DistributionModels/weibull.py)

### Tools
1. [Concatenate multiple torch datasets](vast/tools/ConcatDataset.py) Useful for openset learning.
2. [Feature Extraction](vast/scripts/FeatureExtractors) to HDF5 file from a specific layer of a pytorch model
3. [Multiprocessing Logger](vast/tools/logger.py)

### Visualization
1. 2D visualization e.g. features from LeNet++
2. 3D visualization for decision planes
3. OSRC curve plotter
4. Histogram of scores

### Evaluation
1. OSRC curve using torch tensors and cuda operations
2. FPR vs coverage plot
3. Precision/Recall for binary class OOD problem
4. F-ß Score

Unless a module has a separate license, this code is free for non-commercial use.  We only ask that you cite one our papers, which everyone is most apprpiriate:


## Examples
Some research works using this repo are:

[Self-Supervised Features Improve Open-World Learning](https://github.com/Vastlab/SSFiOWL)

[MNIST Based Experiments](https://github.com/Vastlab/MNIST_Experiments)

[ImageNet Level Openset experiments](https://github.com/Vastlab/ImageNetDali)
