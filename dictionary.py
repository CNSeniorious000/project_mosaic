import cv2


def inter_area(a):
    return [(a,cv2.INTER_AREA)]

def inter_area_nearest(a, b):
    return [
        (a, cv2.INTER_AREA),
        (b, cv2.INTER_NEAREST)
    ]

rgb = (0, 1, 2)
rbg = (0, 2, 1)
gbr = (1, 2, 0)
grb = (1, 0, 2)
brg = (2, 0, 1)
bgr = (2, 1, 0)
