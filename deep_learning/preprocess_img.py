import cv2, os
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3 # that is the NVidia model input
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_image_from_dir(data_dir, image_file):
    #Load RGB images from a file
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def load_image(image_file):
    #Load RGB images from a file
    return mpimg.imread(image_file.strip())

def crop(image):
    #Crop the image (removing the sky at the top and the car front at the bottom)
    return image[60:-25, :, :]

def resize(image):
	return tf.image.resize(image, IMAGE_SIZE) # images should be that size anyway
    #return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def normalize_image(image):
    return image/127.5 - 1
    #return tf.image.per_image_standardization(image)

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = rgb2yuv(image)
    image = resize(image)
    #image = normalize_image(image)
    return image

def preprocess_with_norm(image):
    """
    Combine all preprocess functions into one
    """
    image = preprocess(image)
    image = normalize_image(image)
    return image
