import math

from base import BaseModule

import torch
import torch.nn
import torch.utils.checkpoint

from datasets.ASDNDataset import interpolating_fn

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation

# define how much checkpoint will be created
# see https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential
# trade off between memory and training time
_SEGMENTS_GRADIENT_CHECKPOINT = 1
_ENABLE_GRADIENT_CHECKPOINT = _SEGMENTS_GRADIENT_CHECKPOINT > 1

def set_save_checkpoints(n : int):
    _SEGMENTS_GRADIENT_CHECKPOINT = n
    _ENABLE_GRADIENT_CHECKPOINT = _SEGMENTS_GRADIENT_CHECKPOINT > 1

# region DenseLayer
class DenseLayer(torch.nn.Module):
    r"""Base **dense layer** where input is concatenated to the output (input forwarded into ``base`` ).

    .. math::

        \textnormal{Given } D_{i-1} \textnormal{ the input of the i-th dense layer and } D_i \textnormal{ the output of the i-th dense layer then }

        D_i = [D_{i-1}, base(D_{i-1})]

        \textnormal{ and so } D_{i+1} = D_i

        \textnormal{ and } D_{i+2} = [D_{i+1},base(D_{i+1})]

        \textnormal{ therefore } D_{i+2} = [D_{i}, base(D_{i+1})].

        \textnormal{ So the output } D_{i+2} \textnormal{ is given by the recursive concatenation of all previous input and output }

         (D_i = [ D_{i-1}, base(D_{i-1}) ] = [ D_{i-2}, base(D_{i-2},base(D_{i-1} )

        \textnormal{ and the current output } ( base(D_{i+1}) )


    NOTE: For a proper way to use torch.utils.checkpoint.checkpoint see:

    - https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

    - https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb

    Args:
      base : The base module used for creating the dense layer
    Attributes:
      base : The base module used for creating the dense layer
    """

    def __init__(self, base: torch.nn.Module):
        super(DenseLayer, self).__init__()
        self.base = base

    def forward(self, input: torch.Tensor):
        def concat_(input):
            return torch.cat([input, self.base(input)], dim=1)

        if _ENABLE_GRADIENT_CHECKPOINT and input.requires_grad:
            return torch.utils.checkpoint.checkpoint(concat_, input)
        else:
            return concat_(input)


# endregion

# region IntraDenseBlock
class IntraDenseBlock(torch.nn.Module):
    """ Base Intra dense block

      Args:
        in_channels: Input channels of the block
        out_compressed_channels: Output channels of the input and output compression convolution at the beginning and at the end of the block
        intra_layer_output_features: Output channels of each convolution inside the block
        n_intra_layers: Number of convolutions inside the block
      """

    def __init__(self, in_channels: int, out_compressed_channels: int, intra_layer_output_features: int,
                 n_intra_layers: int):
        super(IntraDenseBlock, self).__init__()

        self.input_compression_layer = torch.nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_compressed_channels,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0)

        self.output_compression_layer = torch.nn.Conv2d(
            in_channels=out_compressed_channels + n_intra_layers * intra_layer_output_features,
            out_channels=out_compressed_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.n_intra_layers = n_intra_layers

        list_intra_layers = list()
        for n in range(n_intra_layers):
            # e.g:
            # n_intra_layers = 4
            # IC = input compression
            # OC = output compression
            # i:| = i-th Conv
            # IC -> 0:| -> 1:| -> 2:| -> 3:| -> [Here we have as input for OC: n_r + 4*n_g] OC
            intra_layer_input_features = out_compressed_channels + n * intra_layer_output_features

            # A sequence of Conv(kernel_size=3,output_channels=n_g) and Relu
            list_intra_layers.append(
                DenseLayer(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intra_layer_input_features,
                                        out_channels=intra_layer_output_features,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                        torch.nn.ReLU(inplace=True))))

        self.intra_layers = torch.nn.Sequential(*list_intra_layers)

    def forward(self, input: torch.Tensor):
        compressed_input = self.input_compression_layer(input)

        if _ENABLE_GRADIENT_CHECKPOINT and compressed_input.requires_grad:
            dense_output = torch.utils.checkpoint.checkpoint_sequential(self.intra_layers,
                                                                        _SEGMENTS_GRADIENT_CHECKPOINT,
                                                                        compressed_input)
        else:
            dense_output = self.intra_layers(compressed_input)

        compressed_output = self.output_compression_layer(dense_output)

        return compressed_input + compressed_output


# endregion

# region ChannelAttention
class ChannelAttention(torch.nn.Module):
    """Base Channel-Wise Attention module

      Args:
        in_channels: Input channels
        reduction: Reduction to use for compressing channels information ( channels_reduced = in_channels // reduction )
      """

    def __init__(self, in_channels: int, reduction: int):
        super(ChannelAttention, self).__init__()

        channels_reduced = in_channels // reduction

        self.attention = torch.nn.Sequential(
            # max pooling
            torch.nn.AdaptiveMaxPool2d(output_size=1),

            # apply reduction
            torch.nn.Conv2d(in_channels=in_channels, out_channels=channels_reduced, kernel_size=1),

            torch.nn.ReLU(),

            # remove reduction
            torch.nn.Conv2d(in_channels=channels_reduced, out_channels=in_channels, kernel_size=1),

            torch.nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor):
        attention_mask = self.attention(input)
        return input * attention_mask


# endregion

# region DenseAttentionBlock
def DenseAttentionBlock(in_channels: int, out_channels: int, intra_layer_output_features: int, n_intra_layers: int,
                        reduction: int):
    '''Base Dense Attention Block used in the paper

    Args:
      in_channels : Input channels of the Dense Attention Block
      out_channels : Output channels of the Dense Attention Block ( which is equal to out_compressed_channels: output channels of the compression convolutions (at the beginning and at the end of IntraDenseBlock) ( see :class:`~IntraDenseBlock` ) )
      intra_layer_output_features: Output channels of each convolution inside the block  ( see :class:`~IntraDenseBlock` )
      n_intra_layers: Number of convolutions inside the block  ( see :class:`~IntraDenseBlock` )
      reduction: Reduction to use for compressing channels information ( channels_reduced = in_channels // reduction )  ( see :class:`~ChannelAttention` )
    '''

    return torch.nn.Sequential(
        IntraDenseBlock(in_channels=in_channels, out_compressed_channels=out_channels,
                        intra_layer_output_features=intra_layer_output_features, n_intra_layers=n_intra_layers),
        ChannelAttention(in_channels=out_channels, reduction=reduction)
    )


# endregion

# region FeatureMappingBranch
class FeatureMappingBranch(torch.nn.Module):
    """ Feature Extraction Branch
    Args:
      in_channels: Input image channel(s) (1:grayscale, 3:rgb)
      low_level_features: Output channels of the low level features extraction convolution
      n_dab: Number of Dense Attention Blocks
      out_channels_dab: Output channels of the Dense Attention Block ( which is equal to out_compressed_channels: output channels of the compression convolutions (at the beginning and at the end of IntraDenseBlock) ( see :class:`~IntraDenseBlock` ) )
      intra_layers_output_features: Output channels of convolutions inside the Intra Dense Block ( see :class:`~IntraDenseBlock` )
      n_intra_layers: Number of convolutions inside Intra Dense Block ( see :class:`~IntraDenseBlock` )
      reduction: Reduction applied inside the Channel Attention layer ( see :class:`~ChannelAttention` )

    Attributes:
      final_output: the final output channels after the concatenation of all previous channels
    """

    def __init__(self, in_channels, low_level_features, n_dab, out_channels_dab, intra_layer_output_features,
                 n_intra_layers, reduction):
        super(FeatureMappingBranch, self).__init__()

        self.final_output = low_level_features + n_dab * out_channels_dab

        # extract low level features from the input
        self.low_level_features = torch.nn.Conv2d(in_channels=in_channels,
                                                  out_channels=low_level_features,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1)

        # Dense attention blocks
        self.n_dab = n_dab
        dabs = list()
        for n in range(n_dab):
            block_input_channels = low_level_features + n * out_channels_dab
            dabs.append(
                DenseLayer(
                    DenseAttentionBlock(in_channels=block_input_channels,
                                        out_channels=out_channels_dab,
                                        intra_layer_output_features=intra_layer_output_features,
                                        n_intra_layers=n_intra_layers,
                                        reduction=reduction)))

        self.dabs = torch.nn.Sequential(*dabs)

    def forward(self, input):
        low_level_features = self.low_level_features(input)
        if _ENABLE_GRADIENT_CHECKPOINT and low_level_features.requires_grad:
            dense_output = torch.utils.checkpoint.checkpoint_sequential(self.dabs,
                                                                        _SEGMENTS_GRADIENT_CHECKPOINT,
                                                                        low_level_features)
        else:
            dense_output = self.dabs(low_level_features)

        return dense_output


# endregion

# region SpatialAttention
class SpatialAttention(torch.nn.Module):
    def __init__(self, in_channels, expansion):
        super(SpatialAttention, self).__init__()

        self.sa = torch.nn.Sequential(
            # expand the channels for each activation
            torch.nn.Conv2d(in_channels=in_channels, out_channels=expansion * in_channels, kernel_size=1, stride=1,
                            padding=0),

            # extract information at each spatial location
            torch.nn.ReLU(),

            # restore the original channels dimension at each activation
            torch.nn.Conv2d(in_channels=in_channels * expansion, out_channels=in_channels, kernel_size=1, stride=1,
                            padding=0),

            # create the spatial-attention mask
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        mask = self.sa(input)
        return input * mask


# endregion

# region ImageReconstructionBranch
class ImageReconstructionBranch(torch.nn.Module):
    """ Image Reconstruction Branch defined inside the paper
    Args:
      in_channels: Input channels of the Image Reconstruction Branch ( which is the output channels of the Feature Extraction Branch )
      out_channels: Input image channel(s) (1:grayscale, 3:rgb)
      expansion: How much expand the channels inside the Spatial Attention for extracting high level frequency spatial information
    """

    def __init__(self, in_channels, out_channels, expansion):
        super(ImageReconstructionBranch, self).__init__()

        self.transform_into_image = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                    stride=1, padding=0)

        self.sa = SpatialAttention(out_channels, expansion)

    def forward(self, input_image, input_features):
        activations_to_image = self.transform_into_image(input_features)

        high_frequency_spatial_information = self.sa(activations_to_image)

        return input_image + high_frequency_spatial_information


# endregion

class ASDN(BaseModule):
    """ASDN: A Deep Convolutional Network for Arbitrary Scale Image Super-Resolution implementation

    Args:
        input_image_channels : Input image channels
        lfr : Laplacian Frequency Representation object ( initialized in order to hold pyramid levels information )
        **FMB_kwargs: key-value arguments for overriding paper parameters of the Feature Mapping Branch (see :class:`~FeatureMappingBranch`)

    Keyword Args:
        in_channels
        low_level_features
        n_dab
        out_channels_dab
        intra_layer_output_features
        n_intra_layers
        reduction
    """

    def __init__(self, input_image_channels: int, lfr: LaplacianFrequencyRepresentation, **FMB_kwargs):
        super(ASDN, self).__init__()

        self.lfr = lfr

        # Use custom parameters if available otherwise relay on ones defined in the paper
        self.feature_mapping_branch = FeatureMappingBranch(in_channels=input_image_channels,
                                                           low_level_features=FMB_kwargs.get("low_level_features", 64),
                                                           n_dab=FMB_kwargs.get("n_dab", 16),
                                                           out_channels_dab=FMB_kwargs.get("out_compressed_channels",
                                                                                           64),
                                                           intra_layer_output_features=FMB_kwargs.get(
                                                               "intra_layer_output_features", 64),
                                                           n_intra_layers=FMB_kwargs.get("n_intra_layers", 8),
                                                           reduction=FMB_kwargs.get("reduction", 16))

        # change this if you are going to change the feature extractor
        self.in_channels_irb = self.feature_mapping_branch.final_output
        # and also this
        self.image_reconstruction_branches = torch.nn.ModuleList(
            [ImageReconstructionBranch(self.in_channels_irb, input_image_channels, 2) for _ in range(lfr.count)])

    def forward(self, interpolated_patch, irb_index: int):
        # features_extracted = self.feature_mapping_branch(interpolated_patch)
        # output_leveli = self.image_reconstruction_branches[irb_index](interpolated_patch, features_extracted)

        # one line return for avoid unnecessary allocation ?!?
        return self.image_reconstruction_branches[irb_index](interpolated_patch,
                                                             self.feature_mapping_branch(interpolated_patch))

    def train_step(self, scale, low_res_batch_i_minus_1, low_res_batch_i):
        level_i_minus_1, level_i = self.lfr.get_for(scale)

        # get the last size: width
        OUT_SIZE = low_res_batch_i.size(-1)

        # forward level i
        out_level_i = self(low_res_batch_i, level_i.index)

        # forward level i-1
        out_level_i_minus_1 = self(low_res_batch_i_minus_1, level_i_minus_1.index)
        out_level_i_minus_1 = interpolating_fn(out_level_i_minus_1, size=(OUT_SIZE, OUT_SIZE))

        # compute the phase at level i
        phase = out_level_i_minus_1 - out_level_i

        # interpolate for reconstructing image
        # reconstructed_image =
        return out_level_i + self.lfr.get_weight(scale) * phase

    @torch.no_grad()
    def val_step(self, scale, low_res_batch_i_minus_1, low_res_batch_i):
        return self.train_step(scale, low_res_batch_i_minus_1, low_res_batch_i)

    @torch.no_grad()
    def test_step(self, scale, low_res_batch_i_minus_1, low_res_batch_i):
        raise NotImplementedError
