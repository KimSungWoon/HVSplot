import inline as inline
import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image

#%matplotlib inline

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def image_color_cluster(image_path, k=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=k)
    clt.fit(image)

    hist = centroid_histogram(clt)

#   for center in clt.cluster_centers_:
    df = pd.DataFrame(clt.cluster_centers_, columns=list('RGB')) #, columns=list('RGB')
    df.to_csv('crop_data.csv', header=False, index=None, mode='a')

    # bar = plot_colors(hist, clt.cluster_centers_)

    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()

files = [f for f in listdir('/Users/sungwoonkim/Desktop/겨울왕국2.mkv') if
         isfile(join('/Users/sungwoonkim/Desktop/겨울왕국2.mkv', f))]

if __name__ == '__main__':
    for j in files:
        image_color_cluster('/Users/sungwoonkim/Desktop/겨울왕국2.mkv/' + j)
