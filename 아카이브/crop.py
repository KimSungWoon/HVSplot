from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image
import cv2
import numpy as np
import matplotlib.image as mpimg

# files = [f for f in listdir('/content/animation_cut') if isfile(join('/content/animation_cut', f))]

files = [f for f in listdir('/Users/sungwoonkim/Desktop/겨울왕국2.mkv') if
         isfile(join('/Users/sungwoonkim/Desktop/겨울왕국2.mkv', f))]


def image_crop1(infilename, save_path):
    # dir = r"/content/animation_cut/"
    # im = "*.png"
    img_dir = "/Users/sungwoonkim/Desktop/겨울왕국2.mkv"
    img = Image.open(infilename)
    (img_h, img_w) = img.size
    print(img.size)
    grid_w = (int)(img_w / 2)
    grid_h = (int)(img_h / 3)
    print(grid_w, grid_h)

    range_w = (int)(img_w / grid_w)
    range_h = (int)(img_h / grid_h)
    print(range_w, range_h)
    i = 0
    for w in range(range_w):
        for h in range(range_h):
            bbox = (h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
            print(h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
            # 가로 세로 시작, 가로 세로 끝
            crop_img = img.crop(bbox)

            fname = "{}".format("{0:01d}").format(i) + j
            savename = save_path + fname
            crop_img.save(savename)
            print('save file ' + savename)
            i += 1


if __name__ == '__main__':
    k = 0
    for j in files:
        image_crop1('/Users/sungwoonkim/Desktop/겨울왕국2.mkv/' + j, '/Users/sungwoonkim/Desktop/겨울왕국_cut/')
        k += 1