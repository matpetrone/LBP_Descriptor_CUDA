import argparse
import os
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

def padding():
    pass



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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # images_directory
    parser.add_argument("-id", "--img_dir", help="image directory", default="res/images", type=str)
    # filename
    parser.add_argument("-f", "--filename", help="image filename", default="gray.jpg", type=str)
    # output dir
    parser.add_argument("-od", "--out_dir", help="output directory to save generated csv file", default="res/csv_images", type=str)

    args = parser.parse_args()
    main(args.img_dir, args.filename, args.out_dir)
