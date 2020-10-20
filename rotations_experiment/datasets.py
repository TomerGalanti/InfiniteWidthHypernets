from __future__ import print_function
import torch
import random
import torchvision
from torchvision import datasets, transforms
from utils.transforms import Rotate
from PIL import Image



class MNIST_ROTATE(datasets.MNIST):
    def __init__(self, length, **kwargs):
        super(MNIST_ROTATE, self).__init__(**kwargs)
        self.length = length

    @staticmethod
    def load_img(img):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        return img

    def get_image(self, index):
        img = self.data[index]
        img = self.load_img(img)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):

        if self.length == None:
            return super().__len__()
        return self.length


    def __getitem__(self, idx):

        angle1 = random.uniform(0,360)
        angle2 = random.uniform(0,360)

        transform1 = transforms.Compose([
            Rotate(angle1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform2 = transforms.Compose([
            Rotate(angle1+angle2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        img = self.get_image(idx)
        x = transform1(img)
        y = transform2(img)

        angles = [30*i for i in range(12)]

        closest = min(angles, key=lambda x:abs(x-angle2))

        return x, y, closest/30, angle2



class CIFAR_ROTATE(datasets.CIFAR10):
    def __init__(self, length, **kwargs):
        super(CIFAR_ROTATE, self).__init__(**kwargs)

        self.length = length

    @staticmethod
    def load_img(img):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        return img

    def get_image(self, index):
        img = self.data[index]
        img = self.load_img(img)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):

        if self.length == None:
            return super().__len__()
        return self.length

    def __getitem__(self, idx):


        # get a random image
        img = self.get_image(idx)
        z = transforms.ToTensor()(img)

        # two random angles for the rotation
        angle1 = random.uniform(0, 360)
        angle2 = random.uniform(0, 360)

        x = torch.zeros((3,32,32))
        y = torch.zeros((3,32,32))

        for i in range(3): # for each channel
            Img_PIL = torchvision.transforms.functional.to_pil_image(z[i].unsqueeze(0))

            Img_PIL_Rot1 = torchvision.transforms.functional.rotate(Img_PIL, angle1, fill=(0,))
            Img1 = transforms.ToTensor()(Img_PIL_Rot1)
            x[i] = Img1

            Img_PIL_Rot2 = torchvision.transforms.functional.rotate(Img_PIL, angle1+angle2, fill=(0,))
            Img2 = transforms.ToTensor()(Img_PIL_Rot2)
            y[i] = Img2

        # normalize samples
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
        y = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(y)

        angles = [30*i for i in range(12)]

        closest = min(angles, key=lambda x:abs(x-angle2))

        return x, y, closest/30, angle2

