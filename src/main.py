import cv2 as cv
import numpy as np


def drawHistogram() -> None:
    img = cv.imread('./src/image.png')

    h = np.zeros((300, 256, 3))

    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(color):
        hist_item = cv.calcHist([img], [ch], None, [256], [0, 255])
        cv.normalize(hist_item, hist_item, 0, 255, cv.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv.polylines(h, [pts], False, col)

    h = np.flipud(h)

    cv.imshow('colorhist', h)
    cv.waitKey(0)


def main():
    drawHistogram()


main()
