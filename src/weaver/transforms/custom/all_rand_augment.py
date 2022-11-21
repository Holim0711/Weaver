from typing import Optional, List, Tuple
import torch
from PIL.Image import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import _apply_op

__all__ = ['AllRandAugment']


class AllRandAugment:
    r"""Random augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_,
    `"Unsupervised Data Augmentation for Consistency Training"
    <https://arxiv.org/abs/1904.12848>`_,
    `"FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
    <https://arxiv.org/abs/2001.07685>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Maximum number of augmentation transformations to apply sequentially.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_ops = num_ops
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, image_size: Tuple[int, int]):
        return [
            # (name, min, max, signed)
            ("Identity", 0.0, 0.0, False),
            ("ShearX", 0.0, 0.3, True),
            ("ShearY", 0.0, 0.3, True),
            ("TranslateX", 0.0, 0.3 * image_size[1], True),
            ("TranslateY", 0.0, 0.3 * image_size[0], True),
            ("Rotate", 0.0, 30.0, True),
            ("Brightness", 0.05, 0.95, True),
            ("Color", 0.05, 0.95, True),
            ("Contrast", 0.05, 0.95, True),
            ("Sharpness", 0.05, 0.95, True),
            ("Posterize", 8.0, 4.0, False),
            ("Solarize", 255.0, 0.0, False),
            ("AutoContrast", 0.0, 0.0, False),
            ("Equalize", 0.0, 0.0, False),
        ]

    def __call__(self, img: Image) -> Image:
        op_list = self._augmentation_space(img.size)
        n = torch.randint(2, (self.num_ops,)).sum().item()
        i_list = torch.randint(len(op_list), (n,)).tolist()
        m_list = torch.rand(n).tolist()

        for i, m in zip(i_list, m_list):
            op, a, b, signed = op_list[i]
            m = m * (b - a) + a
            if signed and torch.randint(2, (1,)):
                m *= -1
            img = _apply_op(img, op, m, interpolation=self.interpolation, fill=self.fill)

        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
