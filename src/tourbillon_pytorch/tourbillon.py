# Currently only supports one-layer encoder and decoders

from einops.layers.torch import Rearrange
import numpy as np
import torch
from torch import nn

from collections import OrderedDict
from typing import Union, Tuple, List, OrderedDict


def _preprocess_conv_arg(
    arg: Union[int, Tuple[int, int]], num_blocks: int
) -> List[Union[int, Tuple[int, int]]]:
  """If `arg` is not a list, repeats the argument in a list of length 
  `num_layers`
  
  Args:
    arg: argument of TourbillonBuildingBlockConv constructor
    num_layers: number of TourbillonBuildingBlockConv to construct

  Returns:
    list of arguments for TourbillonBuildingBlockConv constructors
  """

  if type(arg) in (int, tuple):
    arg = [arg] * num_blocks
  assert len(arg) == num_blocks, 'Number of conv args exceeds number of blocks'
  return arg


class TourbillonBuildingBlockBase(nn.Module):
  """Circular autoencoder
  
  Args:
    encoder: encoder of autoencoder
    decoder: decoder of autoencoder
    num_circulations: how many times to cycle through the autoencoder
    target_circulation: which circulation to take targets from
    output_circulation: which circulation to take outputs from
  """

  def __init__(
      self,
      encoder: nn.Module,
      decoder: nn.Module,
      num_circulations: int = 2,
      target_circulation: int = 0,
      output_circulation: int = 1
  ) -> None:
    super().__init__()
    self.num_circulations = num_circulations
    self.target_circulation = target_circulation
    self.output_circulation = output_circulation

    self.encoder = encoder
    self.decoder = decoder
  
  def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    """Forward pass

    Args:
      x: input tensor

    Returns:
      ordered dictionary with the following content:
        "enc_target": encoder output during target_circulation
        "enc_output": encoder output during output_circulation
        "dec_target": decoder output during target_circulation
        "dec_output": decoder output during output_circulation
    """

    outputs = OrderedDict()
    
    for i in range(self.num_circulations):
      x = self.encoder(x)
      if i == self.output_circulation:
        outputs['enc_output'] = x
      x = x.detach()
      if i == self.target_circulation:
        outputs['enc_target'] = x
      
      x = self.decoder(x)
      if i == self.output_circulation:
        outputs['dec_output'] = x
      x = x.detach()
      if i == self.target_circulation:
        outputs['dec_target'] = x

    return outputs


class TourbillonBuildingBlockLinear(TourbillonBuildingBlockBase):
  """Circular autoencoder with feed-forward layers
  
  Args:
    in_features: number of input features
    hidden_features: number of hidden features to project to
    num_circulations: how many times to cycle through the autoencoder
    target_circulation: which circulation to take targets from
    output_circulation: which circulation to take outputs from
  """

  def __init__(
      self,
      in_features: int,
      hidden_features: int,
      num_circulations: int = 2,
      target_circulation: int = 0,
      output_circulation: int = 1
  ) -> None:
    self.in_features = in_features
    self.hidden_features = hidden_features

    encoder = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.Tanh()
    )
    decoder = nn.Sequential(
        nn.Linear(hidden_features, in_features),
        nn.Sigmoid()
    )

    super().__init__(encoder, decoder, num_circulations, target_circulation, output_circulation)


class TourbillonBuildingBlockConv(TourbillonBuildingBlockBase):
  """Circular cutoencoder with convolutional layers
  
  Args:
    in_features: number of input channels
    hidden_features: number of hidden channels to project to
    kernel_size: encoder and decoder kernel size
    stride: encoder and decoder stride
    padding: encoder and decoder padding
    num_circulations: how many times to cycle through the autoencoder
    target_circulation: which circulation to take targets from
    output_circulation: which circulation to take outputs from
  """

  def __init__(
      self,
      in_channels: int,
      hidden_channels: int,
      kernel_size: Union[int, Tuple[int, int]],
      stride: Union[int, Tuple[int, int]] = 1,
      padding: Union[int, Tuple[int, int]] = 0,
      num_circulations: int = 2,
      target_circulation: int = 0,
      output_circulation: int = 1
  ) -> None:

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels

    encoder = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size, stride, padding),
        nn.Tanh()
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size, stride, padding),
        nn.Sigmoid()
    )

    super().__init__(encoder, decoder, num_circulations, target_circulation, output_circulation)


class TourbillonBase(nn.Module):
  """Stack of circular autoencoders

  Args:
    blocks: list of Tourbillon building blocks
    last_hidden_size: hidden size of the last building block
    classes: Number of classes. Can be set to `0` to return final block output
             instead of class scores.
    neck: module for processing output of blocks before final classifier
  """

  def __init__(
      self,
      blocks: nn.ModuleList,
      last_hidden_size: int,
      classes: int,
      neck: nn.Module = nn.Identity()
  ):
    super().__init__()
    self.blocks = blocks
    self.neck = neck
    self.head = nn.Linear(last_hidden_size, classes) if classes > 0 \
                else nn.Identity()
    if classes > 0:
      nn.init.xavier_uniform_(self.head.weight) #bias too?

  def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    """Forward pass

    Args:
      x: input tensor

    Returns:
      ordered dictionary with the following content:
        "output": final output
        for each block i:
          "enc_target_i": encoder output during block i's target_circulation
          "enc_output_i": encoder output during block i's output_circulation
          "dec_target_i": decoder output during block i's target_circulation
          "dec_output_i": decoder output during block i's output_circulation
    """

    outputs = OrderedDict()
    for i, block in enumerate(self.blocks):
      x = block(x)
      outputs.update({f'{k}_{i}': v for k, v in x.items()})
      x = x['enc_output']
    x = x.detach()
    x = self.neck(x)
    x = self.head(x)
    outputs['output'] = x
    return outputs


class TourbillonLinear(TourbillonBase):
  """Stack of circular autoencoders with feed-forward layers
  
  Args:
    sizes: list consisting of input size followed by hidden size of each block
    classes: Number of classes. Can be set to `0` to return final block output
             instead of class scores.
    num_circulations: how many times to cycle through each autoencoder
    target_circulation: which circulation to take targets from
    output_circulation: which circulation to take outputs from
  """

  def __init__(
      self,
      sizes: List[int],
      classes: int,
      num_circulations: int = 2,
      target_circulation: int = 0,
      output_circulation: int = 1
  ) -> None:
    blocks = nn.ModuleList()
    for in_size, out_size in zip(sizes[:-1], sizes[1:]):
      blocks.append(TourbillonBuildingBlockLinear(in_size, out_size, num_circulations, target_circulation, output_circulation))
    super().__init__(blocks, sizes[-1], classes)


class TourbillonConv(TourbillonBase):
  """Stack of circular autoencoders with convolutional layers
  
  Args:
    input size: size of input image
    channels: list consisting of input channels followed by the hidden channels 
              of each block
    kernel_sizes: list of kernel sizes for each block
    classes: Number of classes. Can be set to `0` to return final block output
             instead of class scores.
    strides: list of strides for each block
    paddings: list of paddings for each block
    num_circulations: how many times to cycle through each autoencoder
    target_circulation: which circulation to take targets from
    output_circulation: which circulation to take outputs from
  """

  def __init__(
      self,
      input_size: Tuple[int, int],
      channels: List[int],
      kernel_sizes: Union[int, List[Union[int, Tuple[int, int]]]],
      classes: Union[int, List[Union[int, Tuple[int, int]]]],
      strides: Union[int, List[Union[int, Tuple[int, int]]]] = 1,
      paddings: Union[int, List[Union[int, Tuple[int, int]]]] = 0,
      num_circulations: int = 2,
      target_circulation: int = 0,
      output_circulation: int = 1
  ) -> None:
    kernel_sizes = _preprocess_conv_arg(kernel_sizes, len(channels) - 1)
    strides = _preprocess_conv_arg(strides, len(channels) - 1)
    paddings = _preprocess_conv_arg(paddings, len(channels) - 1)
    
    curr_size = np.array(input_size)
    layers = nn.ModuleList()
    for in_channels, hidden_channels, kernel_size, stride, padding in zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings):
      curr_size = curr_size + 2 * np.array(padding) - np.array(kernel_size) + 1
      layers.append(TourbillonBuildingBlockConv(in_channels, hidden_channels, kernel_size, stride, padding, num_circulations, target_circulation, output_circulation))
    
    last_hidden_size = channels[-1] * np.product(curr_size)
    neck = Rearrange('b c h w -> b (c h w)')

    super().__init__(layers, last_hidden_size, classes, neck)