# Probabilistic Inference for Learning Control (PILCO)

We use a `TensorFlow v2` implementation of the [PILCO](https://ieeexplore.ieee.org/abstract/document/6654139/) Algorithm obtained and modified from [here](https://github.com/nrontsis/PILCO).
## The Cart-Pole Swing-Up Example

Since OpenAI gym does not contain the cart-pole swing-up example, we use a third-party implementation obtained and modified from [here](https://github.com/jfpettit/cartpole-swingup-envs).

## Usage

Everything has been tested in a conda environment with `python==3.7`. It is recommended to create a new conda environment for this:

First install the PILCO TensorFlow v2 implementation by:
```
git clone https://github.com/Wenlin-Chen/PILCO
cd PILCO
pip install -e .
pip install -r requirements.txt
```
Then install the cart-pole swing-up environment by:
```
cd examples
pip install -e cartpole-swingup-envs
```
Finally run the cart-pole swing-up example in 'cartpole_swingup.ipynb'.

## Credits:

* [Nikitas Rontsis](https://github.com/nrontsis)
* [Kyriakos Polymenakos](https://github.com/kyr-pol/)
* [Xingdong Zuo](https://github.com/zuoxingdong)
* [Jacob Pettit](https://github.com/jfpettit)
* [Wenlin Chen](https://github.com/Wenlin-Chen)

## References

See the following publications for a description of the algorithm: [1](https://ieeexplore.ieee.org/abstract/document/6654139/), [2](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf), 
[3](https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf)
