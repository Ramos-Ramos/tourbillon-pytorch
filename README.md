# Tourbillon-PyTorch ⌚

Unofficial implementation of [Tourbillon: a Physically Plausible Neural Architecture](https://arxiv.org/abs/2107.06424).

## Disclaimer: not learning?

The forward and backward passes don't throw any errors, but when we train for reconstruction we get gray boxes for MNIST and blue ones for CIFAR-10. Also, when we train for MNIST/CIFAR-10 classification, we get only barely over 10% test accuracy. Perhaps we're implementing the recirculation wrong? If anyone has any ideas on what went wrong, feel free to let us know!

## Installation

```
pip install git+https://github.com/Ramos-Ramos/tourbillon-pytorch
```

## Usage

### Using a Tourbillon Building Block (circular autoencoder)

Using a building block with linear layers
```python
import torch
from tourbillon_pytorch import TourbillonBuildingBlockLinear

# circular autoencoder with linear layers (only two layers currently supported)
model = TourbillonBuildingBlockLinear(
    in_features=784,      # number of input features
    hidden_features=256,  # number of hidden features
    num_circulations=2,   # how many times to cycle through the autoencoder
    target_circulation=0, # which circulation to take targets from
    output_circulation=1  # which circulation to take outputs from
)

input = torch.randn(1, 784)
output = model(input)
loss_enc = torch.nn.functional.mse_loss(output['enc_output'], output['enc_target'])
loss_dec = torch.nn.functional.mse_loss(output['dec_output'], output['dec_target'])
loss = loss_enc + loss_dec
loss.backward()
```

## Citation
```bibtex
@misc{tavakoli2021tourbillon,
      title={Tourbillon: a Physically Plausible Neural Architecture}, 
      author={Mohammadamin Tavakoli and Peter Sadowski and Pierre Baldi},
      year={2021},
      eprint={2107.06424},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
