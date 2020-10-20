import torch
import torchvision.transforms.functional as F
from torchvision import transforms as torch_transforms
import numpy as np
from PIL import ImageOps
from PIL import Image
import math
from numpy import random
import warnings


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, expand=False, pad=False): # pad = False CIFAR
        if pad and self.angle % 90 != 0:
            w, h = img.size
            # # deterimne crop size (without cutting the image)
            # nw, nh = F.rotate(img, self.angle, expand=True).size

            rad_angle = np.deg2rad(self.angle)
            dw = np.abs(np.ceil(w * (np.cos(rad_angle) * np.sin(rad_angle)))).astype(int)
            dh = np.abs(np.ceil(h * (np.cos(rad_angle) * np.sin(rad_angle)))).astype(int)
            img = F.pad(img, padding=(dw, dh), padding_mode='reflect')

            # actual rotation
            img = F.rotate(img, self.angle, fill=(0,))
            #img = F.center_crop(img, (nw, nh))
            #img = F.center_crop(img, (w, h)) # no remove for CIFAR
        else:
            img = F.rotate(img, self.angle, expand=expand, fill=(0,))

        return img


class FlipAndRotate(Rotate):
    def __init__(self, angle, flip):
        super(FlipAndRotate, self).__init__(angle)
        self.flip = flip

    def __call__(self, img, **kwargs):
        if self.flip:
            img = F.hflip(img)

        return super(FlipAndRotate, self).__call__(img, **kwargs)


class RandomAffineRandomFillcolor(torch_transforms.RandomAffine):
    def __init__(self, *args, **kwargs):
        super(RandomAffineRandomFillcolor, self).__init__(*args, **kwargs)
        self.fillcolor = None

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        fillcolor = tuple(np.random.choice(255, len(img.getbands())))
        return F.affine(img, *ret, resample=self.resample, fillcolor=fillcolor)


class RandomTranslateAndResize(object):
    def __init__(self, translate, size):
        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        for t in translate:
            if not (0.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
        self.translate = translate
        self.size = size

    def get_params(self, img_size):
        max_dx = self.translate[0] * img_size[0]
        max_dy = self.translate[1] * img_size[1]
        translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                        np.round(np.random.uniform(-max_dy, max_dy)))

        w = img_size[0] - np.abs(translations[0])
        h = img_size[1] - np.abs(translations[1])

        i = 0 if translations[1] < 0 else translations[1]
        j = 0 if translations[0] < 0 else translations[0]

        return i, j, h, w

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(img.size)
        return F.resized_crop(img, *ret, self.size)


class InvertColors(object):
    """
    Expecting PIL image
    """
    def __call__(self, img):
        return ImageOps.invert(img)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h - self.length)
            x = np.random.randint(w - self.length)

            # y1 = np.clip(y - self.length // 2, 0, h)
            # y2 = np.clip(y + self.length // 2, 0, h)
            # x1 = np.clip(x - self.length // 2, 0, w)
            # x2 = np.clip(x + self.length // 2, 0, w)

            mask[y: y + self.length, x: x + self.length] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomResize(object):
    def __init__(self, ratio=(3. / 4., 4. / 3.)):
        self.ratio = ratio

    def __call__(self, img):
        target_area = img.size[0] * img.size[1]
        aspect_ratio = np.random.uniform(*self.ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        return F.resize(img, (h, w))


class RandomStridedResizedCrop(torch_transforms.RandomResizedCrop):
    """Crop at given stride the given PIL Image to random size and aspect ratio.

        A crop of random size (default: of 0.08 to 1.0) of the original size and a random
        aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
        is finally resized to given size.
        This is popularly used to train the Inception networks.

        Args:
            size: expected output size of each edge
            scale: range of size of the origin size cropped
            ratio: range of aspect ratio of the origin aspect ratio cropped
            interpolation: Default: PIL.Image.BILINEAR
        """

    def __init__(self, size, stride=1, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.stride = stride

    @staticmethod
    def get_params(img, stride, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.choice(np.arange(0, img.size[1] - h, stride))
                j = random.choice(np.arange(0, img.size[0] - w, stride))
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = random.choice(np.arange(0, img.size[1] - w, stride))
        j = random.choice(np.arange(0, img.size[0] - w, stride))
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.stride, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomChoice(torch_transforms.RandomChoice):
    """Apply single transformation randomly picked from a list with given distribution
    """
    def __init__(self, transforms, p=None):
        super(RandomChoice, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        t = random.choice(self.transforms, p=self.p)
        return t(img)
