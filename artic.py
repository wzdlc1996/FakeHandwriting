"""
Combine the content and background
"""
import fakeChar as hw
import PIL
import re
import random


gener = hw.char2imgFromFont()
A4size = (210, 297)
A4hwratio = 297 / 210


def _genImgsForChars(chars, gen=gener):
    res = []
    for x in chars:
        img, (w, h) = gen(x)
        res.append({"img": img, "width": w, "height": h})
    return res


class _lag_artic:
    """
    Handle the article with homogeneous char set
    """
    def __init__(self, filename="./content.md"):
        with open(filename, "r", encoding="utf8") as f:
            """
            self.main should be text list separated by the paragraph. The title is the first element
            """
            self.main_proto = f.readlines()
            self.main_proto = [x[:-1] for x in self.main_proto if x != "\n"]

        self.main = {
            "title": self.main_proto[0],
            "paras": self.main_proto[1:]
        }
        self.chars = re.findall(r"[^\n]", "".join(self.main_proto))
        self.charNum = len(self.chars)
        self.paraNum = len(self.main["paras"])

    def splitLines(self, lineMaxCharNum, paraIndent=True):
        lines = [list(self.main["title"]), [""]]
        for x in self.main["paras"]:
            if paraIndent:
                para = "  " + x
            else:
                para = x
            lines += [list(para)[i:i+lineMaxCharNum] for i in range(0, len(para), lineMaxCharNum)]
        return lines

    def splitPages(self, lineWid: int, lineNum: int, paraIndent=True, titleCent=True) -> list:
        tit = list(self.main["title"])
        if titleCent:
            added = int((lineWid - len(tit)) / 2)
            tit = ["  "] * added + tit + ["  "] * added
        lines = [tit, [""]]
        for x in self.main["paras"]:
            if paraIndent:
                para = "  " + x
            else:
                para = x

            lines += [list(para)[i:i+lineWid] for i in range(0, len(para), lineWid)]
        lines = [lines[i:i+lineNum] for i in range(0, len(lines), lineNum)]
        return lines


class artic:
    """
    Handle the article with real-time generated char set
    """
    def __init__(self, filename="./content.md", gen=gener):
        with open(filename, "r", encoding="utf8") as f:
            """
            self.main should be text list separated by the paragraph. The title is the first element
            """
            self.main_proto = f.readlines()
            # Delete empty line and remove all space
            self.main_proto = [x[:-1].replace(" ", "") for x in self.main_proto if x != "\n"]

        self.gen = gen
        self.charHeight = self.gen.getHeight()
        self.charMWidth = self.gen.getWidth()

        self.main = {
            "title": self.main_proto[0],
            "paras": self.main_proto[1:]
        }
        self.char = {
            "title": _genImgsForChars(self.main["title"], self.gen),
            "paras": [_genImgsForChars(x, self.gen) for x in self.main["paras"]]
        }
        self.paraNum = len(self.main["paras"])
        self.bkpt = [0, 0]

    def make(self, boxSize, dSize, pageOffSet=None):
        res = []
        i = 0
        while self.bkpt[0] < self.paraNum and len(res) < 10:
            iniPage = i == 0
            print(self.bkpt)
            res.append(self.makePage(boxSize=boxSize, dSize=dSize, iniPage=iniPage, pageOffSet=pageOffSet))
            i += 1
        return res

    def makePage(self, boxSize, dSize, iniPage=True, pageOffSet=None):
        pw, ph = boxSize
        dw, dh = dSize

        def resetPos():
            if pageOffSet is None:
                return 0, 0
            else:
                return pageOffSet

        x, y = resetPos()

        # Gen empty image
        img = PIL.Image.new("RGBA", boxSize, None)
        if iniPage:
            self.bkpt = [0, 0]
            titleLen = (len(self.main["title"]) - 1) * dw + sum([x["width"] for x in self.char["title"]])
            x += int((pw - titleLen) / 2)
            for ch in self.char["title"]:
                chImg = ch["img"]
                img.paste(chImg, (x, y), chImg)
                x += ch["width"] + dw
            x, y = resetPos()

            # An empty line below the title
            y += (self.charHeight + dh) * 2

        np, nc = self.bkpt
        while np < self.paraNum:
            while nc < len(self.char["paras"][np]):
                if nc == 0:
                    x += 2 * self.charMWidth
                ch = self.char["paras"][np][nc]
                img.paste(ch["img"], (x, y), ch["img"])
                nc += 1

                x += ch["width"] + dw
                if x >= pw - self.charMWidth:
                    x, _ = resetPos()
                    y += self.charHeight + dh

                if y > ph - self.charHeight:
                    self.bkpt = [np, nc]
                    return img

            x, _ = resetPos()
            y += self.charHeight + dh
            nc = 0
            np += 1

        self.bkpt = [np, nc]
        return img


def loadBackGround(filename=None):
    if filename is None:
        filename = "./background/pku_background.png"
    img = PIL.Image.open(filename)
    img = img.convert("RGBA")
    return img


def intBackGround(page, bk, pos):
    size = bk.size
    abspos = (int(size[0] * pos[0]), int(size[1] * pos[1]))
    img = bk.copy()
    img.paste(page, abspos, page)
    return img


def A4sizer(wid):
    return wid, int(wid * A4hwratio)


def charNumInEachLine(page_w, char_w, dw):
    return int((page_w + dw) / (char_w + dw))