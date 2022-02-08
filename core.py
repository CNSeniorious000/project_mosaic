import cv2, imageio, methods
from loguru import logger


def bbox2xxyy(bbox):
    x, y, w, h = bbox
    return x, x + w, y, y + h

def xxyy2bbox(xxyy):
    x1, x2, y1, y2 = xxyy
    return x1, y1, x2 - x1, y2 - y1

@logger.catch()
def mosaic_inplace(image, xxyy, methods):
    x1, x2, y1, y2 = xxyy
    img = image[y1:y2, x1:x2]
    for factor, algorithm in methods:
        w, h, _ = img.shape
        cv2.resize(img, (w//factor,h//factor), img, interpolation=algorithm)
    image[y1:y2, x1:x2] = cv2.resize(img, (x2-x1,y2-y1), interpolation=cv2.INTER_NEAREST)
    return image  # to enable chain call

def show(image):
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()

rgb = (0, 1, 2)
rbg = (0, 2, 1)
gbr = (1, 2, 0)
grb = (1, 0, 2)
brg = (2, 0, 1)
bgr = (2, 1, 0)

def shift(image, xxyy, rgb=gbr):
    x1, x2, y1, y2 = xxyy
    image[y1:y2, x1:x2] = image[y1:y2, x1:x2, rgb]


""" future features
1. 目前是默认拉伸, 将来可以支持裁切
2. bbox其实不如直接提供x1x2y1y2索引直观, 待下次重构
"""  # TODO 👆



filename = "testcases/2.jpg"



if __name__ == '__main__':
    img = imageio.imread(filename)
    mosaic_inplace(img, (30, 130, 200, -150), methods.area_nearest(5,4))
    mosaic_inplace(img, (175, -175, 100, 175), methods.area(15))
    mosaic_inplace(img, (300, 440, 1820, 1880), methods.area(15))
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()
    # imageio.imwrite(f"{filename[:filename.rindex('.')]}_with_mosaic.png", img)
