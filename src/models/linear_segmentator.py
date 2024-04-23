
import torch
from typing import Tuple

class SegmentationHead(torch.nn.Module):
    def __init__(
            self, 
            token_embedding_dimension: int, 
            patch_size: Tuple[int, int, int], 
            linear_depth: int, 
            num_classes: int):
        """Linear segmentation head for Vision Transformer (inspired by DINOv2 paper)

        Args:
            token_embedding_dimension (int): output dimension of the token embedding from the Vision Transformer
            patch_size (Tuple[int, int, int]): size of the patch in the image (depth, height, width)
            linear_depth (int): number of linear layers in the head (excluding the first projection layer)
            num_classes (int): number of classes in the segmentation task
        """
        self.token_embedding_dimension = token_embedding_dimension
        self.patch_size = patch_size
        self.linear_depth = linear_depth

        patch_depth, patch_height, patch_width = self.patch_size
        linear_output_dimension = patch_height * patch_width * patch_depth * num_classes
        self.projection_layer = torch.nn.Linear(self.token_embedding_dimension, linear_output_dimension)
        self.mlp = [
            torch.nn.Linear(linear_output_dimension, linear_output_dimension) for _ in range(self.linear_depth)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the segmentation head

        Args:
            x (torch.Tensor): output tokens from the Vision Transformer of shape (B, HW/Pw/Pw/Pd, token_embedding_dimension)

        Returns:
            torch.Tensor: segmentation mask of shape (B, Pw, Ph, Cls, Pd)
        """
        x = self.projection_layer(x)
        x = torch.nn.functional.gelu(x)
        for i in range(self.linear_depth):
            x = self.mlp[i](x)
            if i < self.linear_depth - 1:
                x = torch.nn.functional.gelu(x)

        x = self.patchs_to_mask(x)
        x = torch.nn.functional.softmax(x, dim=3) # along the classes channel
        return x

    def patchs_to_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the patches to image

        Args:
            x (torch.Tensor): tensor of shape (B, HW/Pw/Pw/Pd, Pw*Ph*Pd*Cls) to be converted to (B, Pw, Ph, Cls, Pd)

        Returns:
            torch.Tensor: tensor of shape (B, Pw, Ph, Cls, Pd) as a segmentation mask
        """
        patch_depth, patch_height, patch_width = self.patch_size
        x = x.view(-1, patch_height, patch_width, self.num_classes, patch_depth)
        return x