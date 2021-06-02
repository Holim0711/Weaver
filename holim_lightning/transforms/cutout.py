import PIL.ImageDraw
from random import random


def cutout(img, v, fillcolor='black'):
    w, h = img.size
    xc = random()
    yc = random()
    x0 = int(max(0, w * (xc - (v / 2))))
    y0 = int(max(0, h * (yc - (v / 2))))
    x1 = int(min(w, w * (xc + (v / 2))))
    y1 = int(min(h, h * (yc + (v / 2))))
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), fillcolor)
    return img


class Cutout:
    def __init__(self, ratio, fillcolor='black'):
        self.ratio = ratio
        self.fillcolor = fillcolor

    def __call__(self, img):
        return cutout(img, self.ratio, self.fillcolor)
