# Adaptive GAN: Customizing Generators for Adversarial Examples

The implementation of this paper https://arxiv.org/abs/1806.10496

## Usage

1. Pretrain the model of the attack target  with the notebook "Pretrain Target ...."
2. Train the generator to produce adversarial examples with the notebook "Adaptive GAN ..." 


## Requirements
+ tensorflow 1.13
+ numpy
+ tqdm




## References
+ The **Spectral Normalization layers** and **dataset input helpers** are adapted from : https://github.com/minhnhat93/tf-SNDCGAN
