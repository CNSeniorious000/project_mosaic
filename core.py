import imageio
import cv2
from loguru import logger


def bbox2xxyy(bbox):
    x, y, w, h = bbox
    return x, x + w, y, y + h

def xxyy2bbox(xxyy):
    x1, x2, y1, y2 = xxyy
    return x1, y1, x2 - x1, y2 - y1

@logger.catch()
def do_mosaic(img, xxyy, size, down=cv2.INTER_AREA, up=cv2.INTER_NEAREST_EXACT):
    x, y, w, h = xxyy
    tmp = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = cv2.resize(
        cv2.resize(
            tmp, size, interpolation=down
        ), (w, h), interpolation=up
    )

def robust_mosaic(img, rect, factor, down=cv2.INTER_AREA, up=cv2.INTER_NEAREST_EXACT):
    x, y, w, h = rect
    if w <= 0:
        print(f"modified width from {w}", end=" ", flush=True)
        w = img.shape[1] - x + w
        print(f"to {w}.")
    if h <= 0:
        print(f"modified height from {h}", end=" ", flush=True)
        h = img.shape[0] - y + h
        print(f"to {h}.")

    width = w // factor
    height = h // factor

    do_mosaic(img, (x, y, w, h), (width, height), down, up)


def shift(img, x1, x2, y1, y2):
    img[y1:y2, x1:x2] = img[y1:y2, x1:x2, (1,2,0)]


""" future features
1. 目前是默认拉伸, 将来可以支持裁切
2. bbox其实不如直接提供x1x2y1y2索引直观, 待下次重构
"""  # TODO 👆


filename = "testcases/2.jpg"

if __name__ == '__main__':
    # from matplotlib import pyplot as plt
    img = imageio.imread(filename)
    for i in range(10):
        robust_mosaic(img, (30, 200, 100, -150), factor=4, down=cv2.INTER_AREA)
        robust_mosaic(img, (30, 200, 100, -150), factor=5, down=cv2.INTER_LINEAR)
    robust_mosaic(img, (30, 200, 100, -150), factor=20, down=cv2.INTER_AREA)

    robust_mosaic(img, (175, 100, -175, 75), factor=15)
    robust_mosaic(img, (300, 1820, 140, 60), factor=15)
    # plt.imshow(img)
    # plt.show()
    imageio.imwrite(f"{filename[:filename.rindex('.')]}_with_mosaic.png", img)
