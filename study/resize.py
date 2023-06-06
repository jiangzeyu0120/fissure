from PIL import Image
import numpy as np
import cv2
import imageio
# image = Image.open('input_0.jpg')
# low_image = image.resize((256, 256))
# image_black_white = low_image.convert('1')
# image_black_white.save('input_0_new.jpg')

# img = cv2.imread("input_1.jpg")
# img = cv2.resize(img, (256,256))
# img = np.mean(img, axis=2)
# print(img.shape)
# cv2.imwrite('input1_resize.jpg', img)

import tensorflow as tf
img = tf.io.read_file('input_1.jpg')
img = tf.image.decode_png(img, channels=3)
img = tf.image.rgb_to_grayscale(img)
img = tf.image.encode_jpeg(img, quality=100, format='grayscale')
writer = tf.io.write_file('input1_gray.jpg', img)

img = cv2.imread("input1_gray.jpg")
img = cv2.resize(img, (256,256))
cv2.imwrite('input1_resize.jpg', img)

