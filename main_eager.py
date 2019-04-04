import tensorflow as tf
from pathlib import Path as path
import os

tf.enable_eager_execution()

# pro_path = path()
# image_path = pro_path/'wiki_crop'/'00'
# image_root = list(image_path.glob('**/*.*'))
# image_root = [str(i) for i in image_root]

root = os.getcwd()
root += '\\wiki_crop\\00'
img_path = []
for root, dirs, files in os.walk(root):
    for files_name in files:
        img_path.append(root+'\\'+files_name)

def decode_img(path):
    img_raw = tf.read_file(path)
    img_tensor = tf.io.decode_image(img_raw, channels=3)
    img_resize = tf.image.resize_images(img_tensor, [192, 192])
    # print(repr(img_raw)[:100]+"...")
    img_resize /= 255.0
    return img_resize

img_tensors = []
for i in img_path:
    img_tensors.append(decode_img(i))

img_tensors = img_tensors[:100]
img_dataset = tf.data.Dataset.from_tensor_slices(img_tensors)

# def preprocess_image(image):
#   image = tf.image.decode_jpeg(image, channels=3)
#   image = tf.image.resize_images(image, [192, 192])
#   image /= 255.0  # normalize to [0,1] range

#   return image

# image1 = tf.read_file(img_path[0])
# def load_and_preprocess_image(path):
#   image = tf.read_file(path)
#   return preprocess_image(image)

# path_ds = tf.data.Dataset.from_tensor_slices(img_path)
# image_dataset = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
for n,image in enumerate(img_dataset.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(img_path[n])
 

import mat
# 组合img_dataset和mat
# 1.根据img_path重新排mat中age顺序
age_dataset=[]
for file_path in img_path:
    file_path_sliced = file_path[51:]
    for age in mat.age:
        if age.get('filename') == file_path_sliced:
            age_dataset.append(int(age.get('age')))

# 检查一下数据顺序是否正确
# for age in mat.age:
#     if age.get('filename') == '10110600_1985-09-17_2012.jpg':
#         print(age['age'])
age_dataset = age_dataset[:100]
age_dataset = tf.data.Dataset.from_tensor_slices(age_dataset)
image_label_ds = tf.data.Dataset.zip((img_dataset, age_dataset))



a=0
