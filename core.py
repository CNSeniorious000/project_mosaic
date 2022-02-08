import imageio
from loguru import logger
from functools import partial
from dictionary import *


def bbox2xxyy(bbox):
    x, y, w, h = bbox
    return x, x + w, y, y + h

def xxyy2bbox(xxyy):
    x1, x2, y1, y2 = xxyy
    return x1, y1, x2 - x1, y2 - y1

def parse_minus(image, xxyy):
    x1, x2, y1, y2 = xxyy
    h, w = image.shape[:2]
    return x1, (x2-1)%w+1, y1, (y2-1)%h+1

@logger.catch()
def mosaic_inplace(image, xxyy, methods):
    x1, x2, y1, y2 = parse_minus(image, xxyy)
    img = image[y1:y2, x1:x2]
    for factor, algorithm in methods:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w//factor,h//factor), interpolation=algorithm)
    image[y1:y2, x1:x2] = cv2.resize(img, (x2-x1,y2-y1), interpolation=cv2.INTER_NEAREST)
    return image  # to enable chain call

def show(image):
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()

def shift(image, xxyy, rgb=grb):
    x1, x2, y1, y2 = parse_minus(image, xxyy)
    image[y1:y2, x1:x2] = image[y1:y2, x1:x2, rgb]
    return image  # to enable chain call

class WeChatScreenShot(imageio.core.Array):
    def __init__(self, _):
        imageio.core.Array.__init__(self)
        self.mosaic = partial(mosaic_inplace, self)
        self.shift = partial(shift, self)
        self.show = partial(show, self)

    @classmethod
    def imread(cls, path):
        return cls(imageio.imread(path))

    def imwrite(self, path):
        imageio.imwrite(f"{path[:path.rindex('.')]}_with_mosaic.png", self)
        return self

    @property
    def icons(self):
        return 30, 130, 200, -150

    @property
    def title(self):
        return 175, -175, 100, 175

    @staticmethod
    def hit_message(y1, x1, x2=None):
        return x1, x1 + 140 if x2 is None else x2, y1, y1 + 60

    def shift_and_mosaic_icons(self, rgb=grb, area=5, nearest=2):
        self.mosaic(self.icons, inter_area_nearest(area, nearest))
        shift(self, self.icons, rgb)
        return self

    def mosaic_title(self, area=15):
        self.mosaic(self.title, inter_area(area))
        return self

    def mosaic_hit_message(self, y1, x1, x2=None, area=10, nearest=2):
        self.mosaic(self.hit_message(y1,x1,x2), inter_area_nearest(area,nearest))
        return self


if __name__ == '__main__':
    # doing some test
    (
        WeChatScreenShot
        .imread("testcases/2.jpg")
        .mosaic_title()
        .shift_and_mosaic_icons()
        .mosaic_hit_message(x1=300, y1=1810)
        .imwrite("testcase/2.jpg")
        .show()
    )
