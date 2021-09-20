from PIL import Image


def contain_resize(img, size, interpolation=Image.BILINEAR, fillcolor='black'):
    x, y = img.size

    if x > y:
        x, y = size, size * y // x
    else:
        x, y = size * x // y, size

    img = img.resize((x, y), interpolation)

    new = Image.new('RGB', (size, size), fillcolor)

    new.paste(img, ((size - x) // 2, (size - y) // 2))

    return new


class ContainResize():
    def __init__(self, size, interpolation=Image.BILINEAR, fillcolor='black'):
        self.size = int(size)
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    def __call__(self, img):
        return contain_resize(img, self.size, self.interpolation, self.fillcolor)
