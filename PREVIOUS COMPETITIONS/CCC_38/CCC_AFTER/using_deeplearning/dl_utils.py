import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# takes as input directory with images and returns numpy array
def load_images(path_to_img):
    # get all images from path
    img_list = os.listdir(path_to_img)
    img_list.sort()
    # create empty list for numpy images
    img_list_np = []
    # loop over all images
    for img in img_list:
        # open image
        img_open = Image.open(path_to_img + '/' + img)
        # convert image to numpy array
        img_np = np.asarray(img_open)
        # append numpy image to list
        img_list_np.append(img_np)

    # return list of numpy images
    return np.array(img_list_np)

# function that resized images to 64x64
def resize_images(img, size=(64,64)):
    # create empty list for resized images
    img_resized = []
    # loop over all images
    for i in range(len(img)):
        # resize image to 64x64
        img_resized.append(np.array(Image.fromarray(img[i]).resize(size)))
    # return list of resized images
    return np.array(img_resized)

# clears clouds from images
'''
The idea here is to replace the pixels that are white [RGB(255, 255, 255)]
in the first picture (sample0), with the corresponding pixels in the second
picture (sample1). 

A redundant function was initially applied, to replace all white pixels
(in case there were overlapping clouds) with green pixels:

def replace_white(img_array):
    mask = (img_array == [255, 255, 255]).all(axis=3)
    img_array[mask] = np.ones_like(img_array[mask]) * [77, 192, 52]
    return img_array
'''
def clear_clouds(img1, img2, cloud_color=[255, 255, 255]):
    # create mask of cloud pixels in img1
    cloud_mask = np.all(img1 == cloud_color, axis=2)
    # create copy of img1 with cloud pixels replaced by img2 pixels
    img1_copy = np.copy(img1)
    img1_copy[cloud_mask] = img2[cloud_mask]
    # return the modified image
    return img1_copy

# function that draws a circle of given color on an image, without using cv2
def draw_circle(img, center, radius, color):
    # create copy of image
    img_copy = np.copy(img)
    # get image dimensions
    img_h, img_w, _ = img_copy.shape
    # get center coordinates
    x_center, y_center = center
    # loop over all pixels in image
    for i in range(img_h):
        for j in range(img_w):
            # check if pixel is within circle
            if (i - x_center)**2 + (j - y_center)**2 <= radius**2:
                # set pixel color
                img_copy[i, j] = color
    # return modified image
    return img_copy