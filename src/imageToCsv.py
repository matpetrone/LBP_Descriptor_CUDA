import argparse
import os
import csv

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from numpy import savetxt


def rgb2gray(rgb):
    # binary image
    if rgb[..., :3].max() <= 1:
        rgb = rgb[..., :3] * 255
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    # gray = np.array([[min(255, max(0, ai)) for ai in c] for c in gray])
    return gray


# -----------
""" read csv from file """


def loadData(source_dir, skip_header=False):
    """
    Load dataset from file
    :param source_dir: directory where to load data
    :return: loaded dataset
    """
    # d = np.load(source_dir + '.csv')
    dates = []

    with open(source_dir) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        if skip_header: next(csvReader)
        for row in csvReader:
            dates.append([float(r) for r in row])

    return np.array(dates)


""" Plot histogram """

def plotHistogram(x,y, filename):
    # import numpy as np
    # import matplotlib.pyplot as plt

    # n, bins, patches = plt.hist(x, 200, facecolor='g', alpha=0.75, rwidth=30) #, density=True)
    n_pixels = sum(y)
    plt.bar(x, [yi/n_pixels for yi in y])

    plt.xlabel('Pixel value')
    plt.ylabel('Frequency (% of pixels)')
    plt.title('LBP Histogram')
    plt.xlim(0, 256)
    # plt.ylim(0, .5)
    # plt.grid(True)
    plt.savefig(filename)
    plt.show()


image_name = "computer_programming"
histogram_csv_filename = "../res/histograms/" + image_name + "_hist.csv"
data = loadData(histogram_csv_filename)

n_bin = data[:,0]
bin_values = data[:,1]
# print(max(bin_values))

plotHistogram(n_bin, bin_values, "../res/histograms/"+ image_name + ".jpg")


def main(images_dir, filename, out_dir):
    save_gray_image = True
    # file_name = "/home/francesca/PycharmProjects/exampleImage/square.png"
    image_filename = os.path.join(images_dir, filename)
    # file_name = "/home/francesca/PycharmProjects/exampleImage/gray.jpeg"
    print(image_filename)

    image = img.imread(image_filename)
    print(len(image.shape))
    # RGB image
    if len(image.shape) == 3 and (image.shape[-1] == 3 or image.shape[-1] == 4):
        image = rgb2gray(image)

    if save_gray_image:
        imgplot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        # plt.savefig()
        plt.waitforbuttonpress()

        # imgg = Image.fromarray(image)
        # imgg.save("file.png")
    image = image.astype(int)
    # for i in image:
    #     print(i)

    print(image.shape)
    # print(image.shape)
    arr = np.array(image)
    # print(arr.shape)
    # print(max(arr.max(1)), min(arr.min(1)))
    output_file = os.path.join(out_dir, filename.split('.')[0]) + '.csv'
    savetxt(output_file, image, delimiter=',', fmt='%i')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # images_directory
#     parser.add_argument("-id", "--img_dir", help="image directory", default="res/images", type=str)
#     # filename
#     parser.add_argument("-f", "--filename", help="image filename", default="gray.jpeg", type=str)
#     # output dir
#     parser.add_argument("-od", "--out_dir", help="output directory to save generated csv file", default="res/csv_images", type=str)
# 
#     args = parser.parse_args()
#     main(args.img_dir, args.filename, args.out_dir)
