## VAST Lab utile
This repository contains some common functionalities used in various works at the VAST lab.    
Some repositories using this repo are:  
[MNIST Based Experiments](https://github.com/Vastlab/MNIST_Experiments)  
[ImageNet Level Openset experiments](https://github.com/Vastlab/ImageNetDali)  
[Self-Supervised Features Improve Open-World Learning](https://github.com/Vastlab/SSFiOWL)


### Setup
Clone with sub modules  
```git clone --recurse-submodules https://github.com/Vastlab/utile.git```

### Loss Functions
1. Entropic Openset loss
2. Objectosphere loss

### Openset Algorithms
1. OpenMax
2. Multimodal OpenMax
3. Extreme Value Machine (EVM)

### Reimplementation of libMR
This repo contains a torch based reimplementation of the [libMR repo](https://github.com/Vastlab/libMR)  
It supports GPU based computation that speeds up the processing considerably, but the weibull parameter computation may have slight variations.