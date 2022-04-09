"""
All functions related to data, including I/O and manipulation
"""
from impts import *

import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict as od
from typing import OrderedDict, Tuple
from dataType import DataItem


CHAR_W = 64
CHAR_SIZE = (CHAR_W, CHAR_W)
FONT_SIZE = 50
FONT_PATH = "./fonts/other"
CHAR_PATH = "./chars/reg_140.txt"
FONT_PROTO = "./fonts/std/STSong.ttf"

TEST_ID = 10
TEST_CHAR_PATH = "./chars/test_16.txt"



def _getFontSet() -> Tuple[OrderedDict[int, str], OrderedDict[int, str]]:
    font_set = od()
    i = 0
    for ft in os.scandir(FONT_PATH):
        font_set[i] = ft.path
        i += 1

    notest = od()
    i = 0
    for j in range(len(font_set)):
        if j == TEST_ID:
            continue
        notest[i] = font_set[i]
        i += 1
    return font_set, notest


FontSet, FontSet_noTest = _getFontSet()


def _getFontByPath(fontpath: str) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(fontpath, FONT_SIZE)


def _getFontById(font_id: int) -> ImageFont.FreeTypeFont:
    return _getFontByPath(FontSet[font_id])


def getCharImage(char: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new(mode="L", size=CHAR_SIZE, color="white")
    draw = ImageDraw.Draw(img)
    w, h = CHAR_SIZE
    fw, fh = font.getsize(char)
    ofw, ofh = font.getoffset(char)  # Font might be possess self-defined offset
    draw.text(((w - fw - ofw)/2, (h - fh - ofh)/2), char, fill=0, font=font)
    return img


class ChineseCharDataset(Dataset):
    def __init__(self):
        self.transform = trf.Compose([
            trf.Resize(CHAR_W),
            trf.ToTensor()
        ])

        self.protoFont = _getFontByPath(FONT_PROTO)
        with open(CHAR_PATH, "r") as f:
            self.charList = [x for x in f.readline()]

    def getFontNumber(self) -> int:
        return len(FontSet_noTest)

    def getCharNumber(self) -> int:
        return len(self.charList)

    def imageToTensor(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)

    def oneHotEnc(self, ind: int, tot: int) -> torch.Tensor:
        r = torch.zeros(tot)
        r[ind] = 1.
        return r

    def __len__(self) -> int:
        # return len(self.charList)
        return self.getCharNumber() * self.getFontNumber()

    def __getitem__(self, item) -> DataItem:
        """
        Get a random tuple of data. Returns as a tuple of
            -  reference_ind, (Tensor), the index of the random chosen style (one-hot)
            -  character_ind, (Tensor), the index of char (one-hot)
            -  ref_char_ind, (Tensor), the index of random char (one-hot)
            -  prototype_img, (Tensor) 1x64x64, is the image of char with prototype style
            -  reference_img, (Tensor) 1x64x64, is the image of random char with a random style, as reference
            -  real_img, (Tensor) 1x64x64, is the image of char with the style, as the target of generator

        :param item: (int)
        :return: tuple
        """
        char = self.charList[item % self.getCharNumber()]
        proto_img = getCharImage(char, self.protoFont)
        refer_ind = random.choice(list(FontSet_noTest.keys()))
        refer_font = _getFontById(refer_ind)
        refer_char_ind = random.randint(0, self.getCharNumber() - 1)
        refer_char = self.charList[refer_char_ind]
        refer_img = getCharImage(refer_char, refer_font)
        real_img = getCharImage(char, refer_font)
        return DataItem(
            self.oneHotEnc(refer_ind, self.getFontNumber()),
            self.oneHotEnc(item % self.getCharNumber(), self.getCharNumber()),
            self.oneHotEnc(refer_char_ind, self.getCharNumber()),
            self.imageToTensor(proto_img),
            self.imageToTensor(refer_img),
            self.imageToTensor(real_img)
        )


def TensorListToImage(img_list, path):
    img = vutils.make_grid(
        img_list,
        padding=2,
        normalize=True
    )

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title("Refer Images")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig(path)
    plt.close()


class TestCharDataset:
    def __init__(self):
        self.proto_font = _getFontByPath(FONT_PROTO)
        self.refer_font = _getFontById(TEST_ID)
        self.refer_char = "厄"

        self.transform = trf.Compose([
            trf.Resize(CHAR_W),
            trf.ToTensor()
        ])

        with open(TEST_CHAR_PATH, "r") as f:
            self.charList = [x for x in f.readline()]

    def __len__(self) -> int:
        return len(self.charList)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        char = self.charList[item]
        return (
            self.transform(getCharImage(char, self.proto_font)),
            self.transform(getCharImage(self.refer_char, self.refer_font))
        )

    def genRefImage(self):
        img_list = [self.transform(getCharImage(char, self.refer_font)) for char in self.charList]
        TensorListToImage(img_list, "./ref.pdf")


if __name__ == "__main__":
    a = getCharImage("我", _getFontById(0))
    ds = ChineseCharDataset()
    # print(ds[0])
    TestCharDataset().genRefImage()