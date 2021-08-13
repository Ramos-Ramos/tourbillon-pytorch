# Tourbillon-PyTorch ⌚

Unofficial implementation of [Tourbillon: a Physically Plausible Neural Architecture](https://arxiv.org/abs/2107.06424).

## Disclaimer: not learning?

The forward and backward passes don't throw any errors, but when we train for reconstruction we get gray boxes for MNIST and blue ones for CIFAR-10. Also, when we train for MNIST classification, we get only barely over 10% test accuracy. Perhaps we're implementing the recirculation wrong? If anyone has any ideas on what went wrong, feel free to let us know!

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
    in_features=784,
    hidden_features=256,
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

Using a building block with convolutional layers

```python
import torch
from tourbillon_pytorch import TourbillonBuildingBlockConv

# circular autoencoder with convolutional layers (only two layers currently supported)
model = TourbillonBuildingBlockConv(
    in_channels=3,
    hidden_channels=6,
    kernel_size=(5, 5),
    stride=(1, 1),
    padding=(2, 2),
    num_circulations=2,
    target_circulation=0,
    output_circulation=1
)

input = torch.randn(1, 3, 32, 32)
output = model(input)
loss_enc = torch.nn.functional.mse_loss(output['enc_output'], output['enc_target'])
loss_dec = torch.nn.functional.mse_loss(output['dec_output'], output['dec_target'])
loss = loss_enc + loss_dec
loss.backward()
```

### Using Tourbillon (stack of circular autoencoders)

Using a Tourbillon with linear layers

```python
import torch
from tourbillon_pytorch import TourbillonLinear

# stack of circular autoencoders with linear layers (only two layers currently supported)
model = TourbillonLinear(
    sizes=[784, 256, 64], # [input size, hidden size, hidden size]
    classes=10,
    num_circulations=2,
    target_circulation=0,
    output_circulation=1
)

input = torch.randn(1, 784)
labels = torch.randint(0, 10, (1,))
output = model(input)
loss_autoencoders = sum(
    torch.nn.functional.mse_loss(output[f'{half}_output_{block}'], output[f'{half}_target_{block}'])
    for half in ('enc', 'dec') for block in range(len(model.blocks))
)
loss_top = torch.nn.functional.cross_entropy(output['output'], labels)
loss = loss_autoencoders + loss_top
loss.backward()
```
Using a Tourbillon with convolutional layers

```python
import torch
from tourbillon_pytorch import TourbillonConv

# stacked circular autoencoder with convolutional layers (only two layers currently supported)
model = TourbillonConv(
    input_size=(32, 32),              # (h, w) of input
    channels=[3, 6, 6],               # [input channels, hidden channels, hidden channels]
    kernel_sizes=[(5, 5), (17, 17)],
    strides=1,                        # same as [(1, 1), (1, 1)]
    paddings=[2, 0],                  # same as [(2, 2), (0, 0)]
    classes=10,
    num_circulations=2,
    target_circulation=0,
    output_circulation=1
)

input = torch.randn(1, 3, 32, 32)
labels = torch.randint(0, 10, (1,))
output = model(input)
loss_autoencoders = sum(
    torch.nn.functional.mse_loss(output[f'{half}_output_{block}'], output[f'{half}_target_{block}'])
    for half in ('enc', 'dec') for block in range(len(model.blocks))
)
loss_top = torch.nn.functional.cross_entropy(output['output'], labels)
loss = loss_autoencoders + loss_top
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
