import cv2


def area(a):
    return [(a,cv2.INTER_AREA)]

def area_nearest(a, b):
    return [
        (a, cv2.INTER_AREA),
        (b, cv2.INTER_NEAREST)
    ]
