'''
# * 文件描述：此文件用于处理图片，由于原数据集中的图片和年龄是分离的，
# * 通过运行这个文件将其组合到一起成为一个tfrecord文件
'''
import tensorflow as tf
from pathlib import Path as path
import os
tf.enable_eager_execution()




'''
# * 第一步：找到所有图片的绝对路径
'''
root = os.getcwd()
root = root+'\\wiki_crop\\'
img_path = []
for root, dirs, files in os.walk(root):
  for directory in dirs:
    directory = root+directory+'\\'
    for files_name in os.walk(directory):
      for file_name in files_name[2]:
        img_path.append(directory+file_name)



'''
# * 第二步:定义tfrecord函数，详情见官网tfrecord文件描述
'''
def save_to_record_file(file_path, age, writer):
  img_raw = open(file_path, 'rb').read()
  img_tensor = tf.io.decode_image(img_raw, channels=3)
  feature = {
    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_tensor.shape[0]])),
    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_tensor.shape[1]])),
    'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_tensor.shape[2]])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[age])),
    'raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    'file_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_path.encode('utf-8')])),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(example.SerializeToString())




'''
# * 第三步：从mat文件中根据文件名对年龄进行一次比对
'''

import config
import mat
from tqdm import tqdm

# **为了方便测试，可以在这里把图片分小一点，默认为600（一共有60000+张图片）
img_path = img_path[config.img_start : config.img_end]

writer = tf.python_io.TFRecordWriter(config.file_name)
for file_path in tqdm(img_path,  ascii=True):
    file_prefix = os.path.abspath('./wiki_crop')
    file_path_sliced = file_path[len(file_prefix)+4:]
    for age in mat.age:
        if age.get('filename') == file_path_sliced:
          save_to_record_file(file_path, int(age.get('age')), writer)
          break
writer.close()




