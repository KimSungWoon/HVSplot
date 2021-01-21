# Histogram and plotly code from
# http://reynoldsalexander.com/3dhistplots.html
# and non-notebook plotting from
# https://plot.ly/python/getting-started-with-chart-studio/#initialization-for-offline-plotting

import bisect
import itertools
import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

data = np.loadtxt('crop_data.csv', unpack=True, delimiter=',', skiprows=1)  #
X = data[0]
Y = data[1]
Z = data[2]

def plot_3d_hist(img, bin_size, model = "rgb"):

    h, w, num_channels = img.shape

    if isinstance(img, np.uint8):
        max_range = 255
    elif isinstance(img, np.uint16):
        max_range = 65535
    elif isinstance(img, np.float):
        max_range = 1.0
    else:
        max_range = np.amax(img)

    # generate indices and histogram
    ranges = ([0, max_range], [0, max_range], [0, max_range])
    bin_sizes = (bin_size, bin_size, bin_size)
    bin_range = np.arange(bin_size)

    # I don't know why this mesh is ordered this way, but it works with the test
    # images in BGR
    ch2, ch1, ch3 = np.meshgrid(bin_range, bin_range, bin_range)

    ch3 = ch3.ravel()*255/bin_size
    ch2 = ch2.ravel()*255/bin_size
    ch1 = ch1.ravel()*255/bin_size

    hist = np.histogramdd(img.reshape((h*w, num_channels)), bins=bin_sizes, range=ranges)[0]
    hist = hist.ravel()
    ch3, ch2, ch1, hist = ch3[hist>0], ch2[hist>0], ch1[hist>0], hist[hist>0]

    # map histogram amounts to marker sizes
    marker_sizes = [1, 5, 10, 16, 25]
    cut_size = int(np.ceil(hist.size/len(marker_sizes)))
    cuts = list(itertools.islice(sorted(hist), cut_size, None, cut_size))
    assign_marker_size = lambda val : marker_sizes[bisect.bisect(cuts, val)]
    hist_marker_sizes = list(map(assign_marker_size, hist))

    # colors from the image
    if "rgb" == model:
        colors = [f'rgb({ch3[i]}, {ch2[i]}, {ch1[i]})' for i in range(len(hist))]
        labels = ("red (x)", "green (y)", "blue (z)")
    elif "hsv" == model:
        colors = ["hsv(%d, %2.2f%%, %2.2f%%)" % (ch3[i] * 360 / 256,
                                                ch2[i] * 100 / 256,
                                                ch1[i] * 100 / 256) for i in range(len(hist))]
        labels = ("hue (x)", "saturation (y)", "value (z)")
    else:
        raise ValueError("Uncoded model: %s" % model)

    # plotly
    scatter = go.Scatter3d(
        x=ch3, y=ch2, z=ch1, mode='markers',
        marker=dict(size=hist_marker_sizes, color=colors, opacity=1))
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=labels[0], range=[0, max_range]),
            yaxis=dict(title=labels[1], range=[0, max_range]),
            zaxis=dict(title=labels[2], range=[0, max_range])),
        margin=dict(r=0, b=0, l=0, t=0))

    return go.Figure(data=[scatter], layout=layout)

def random_dots(h, w, channel):
    return (200 if channel else 0) + (56 if channel else 255) * np.random.rand(h, w)

def random_image(truncate_channels = []):
    """Generate a random image with 3 channels. If truncate_channels is not empty, the
    random distribution will be over the high values of that channel.
    """
    h = w = 256
    img = np.zeros((h, w, 3))

    img[:, :, 0] = random_dots(h, w, 0 in truncate_channels)
    img[:, :, 1] = random_dots(h, w, 1 in truncate_channels)
    img[:, :, 2] = random_dots(h, w, 2 in truncate_channels)

    return img

def main(filepath = os.path.expanduser("~/Downloads/result.html")):

    # # Generate random images with focus on each of the channels
    img_channel0 = random_image([0])
    img_channel1 = random_image([1])
    img_channel2 = random_image([6])

    # for RGB, channels are BGR, and all these work as I expect
    #pio.write_html(plot_3d_hist(img_channel0, 32, model = "rgb"), file = filepath)  # focus on blue
    #pio.write_html(plot_3d_hist(img_channel1, 32, model = "rgb"), file = filepath)  # focus on green
    #pio.write_html(plot_3d_hist(img_channel2, 32, model = "rgb"), file = filepath)  # focus on red

    # For HSV, I couldn't understand it
    pio.write_html(plot_3d_hist(img_channel2, 32, model = "hsv"), file = filepath)

if "__main__" == __name__:
    main()