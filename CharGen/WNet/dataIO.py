"""
All functions related to data, including I/O and manipulation
"""
from impts import *

import os
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


def _getFontSet() -> OrderedDict[int, str]:
    font_set = od()
    i = 0
    for ft in os.scandir(FONT_PATH):
        font_set[i] = ft.path
        i += 1
    return font_set


FontSet = _getFontSet()


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

    def imageToTensor(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)

    def oneHotEnc(self, ind: int, tot: int) -> torch.Tensor:
        r = torch.zeros(tot)
        r[ind] = 1.
        return r

    def __len__(self) -> int:
        return len(self.charList)

    def __getitem__(self, item) -> DataItem:
        """
        Get a random tuple of data. Returns as a tuple of
            -  style, (Tensor), the index of the random chosen style (one-hot)
            -  chari, (Tensor), the index of char (one-hot)
            -  refi, (Tensor), the index of random char (one-hot)
            -  prototype_img, (Tensor) 1x64x64, is the image of char with prototype style
            -  styled_img, (Tensor) 1x64x64, is the image of random char with a random style, as reference
            -  real_img, (Tensor) 1x64x64, is the image of char with the style, as the target of generator

        :param item: (int)
        :return: tuple
        """
        char = self.charList[item]
        proto_img = getCharImage(char, self.protoFont)
        style_ind = random.choice(list(FontSet.keys()))
        style_font = _getFontById(style_ind)
        style_char_ind = random.randint(0, len(self) - 1)
        style_char = self.charList[style_char_ind]
        style_img = getCharImage(style_char, style_font)
        real_img = getCharImage(char, style_font)
        return DataItem(
            self.oneHotEnc(style_ind, len(FontSet)),
            item,
            style_char_ind,
            self.imageToTensor(proto_img),
            self.imageToTensor(style_img),
            self.imageToTensor(real_img)
        )




if __name__ == "__main__":
    a = getCharImage("我", _getFontById(0))
    ds = ChineseCharDataset()
    print(ds[0])