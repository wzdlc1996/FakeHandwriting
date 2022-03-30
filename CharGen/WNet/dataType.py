from typing import NamedTuple, List
from torch import Tensor


class GeneratorOutput(NamedTuple):
    """
    The type for output of generator
        -  r (Tensor), Generated data
        -  lout (Tensor), output from left encoder(for prototype)
        -  rout (Tensor), output from right encoder(for reference)
    """
    r: Tensor
    lout: Tensor
    rout: Tensor


class DiscriminatorOutput(NamedTuple):
    """
    The type for output of discriminator:
        -  r_gan (Tensor), gan classification
        -  r_font (Tensor), font catagory classification
        -  r_char (Tensor), char catagory classification
        -  feat (list[Tensor]), features during conv
    """
    r_gan: Tensor
    r_font: Tensor
    r_char: Tensor
    feat: List[Tensor]

    def __str__(self):
        return f'<DiscriminatorOutput\n\t{self.r_gan}\n\t{self.r_font}\n\t{self.r_char}\n\tfeatlen={len(self.feat)}\n>'


class DataItem(NamedTuple):
    """
    The type for dataitem in dataset
        -  style_ind, (Tensor), the index of the random chosen style (one-hot)
        -  char_ind, (Tensor), the index of char (one-hot)
        -  ref_ind, (Tensor), the index of random char (one-hot)
        -  prototype_img, (Tensor) 1x64x64, is the image of char with prototype style
        -  styled_img, (Tensor) 1x64x64, is the image of random char with a random style, as reference
        -  real_img, (Tensor) 1x64x64, is the image of char with the style, as the target of generator

    To the paper, each data item is a 3-tuple: (x_j^0, x_k^i, x_j^i), which means:
        1.  x_j^0 (prototype_img) is the prototype, 0 is the style, j is the char (char_ind)
        2.  x_k^i (styled_img) is the reference, i is the style (style_ind), k is the char (ref_ind)
        3.  x_j^i (real_img) is the real_inst, i is the style (style_ind), j is the char (char_ind)
    """
    style_ind: Tensor
    char_ind: Tensor
    ref_ind: Tensor
    prototype_img: Tensor
    styled_img: Tensor
    real_img: Tensor

