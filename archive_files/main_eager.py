import tensorflow as tf
from pathlib import Path as path
import os

tf.enable_eager_execution()

'''
# * 第一步：找到所有图片的路径
# ? 使用pathlib模块找相对路径？
# ? 使用os模块找绝对路径
'''
# pro_path = path()
# image_path = pro_path/'wiki_crop'/'00'
# image_root = list(image_path.glob('**/*.*'))
# image_root = [str(i) for i in image_root]

root = os.getcwd()
root = root+'\\wiki_crop\\'
img_path = []
for root, dirs, files in os.walk(root):
  for directory in dirs:
    directory = root+directory+'\\'
    for files_name in os.walk(directory):
      for file_name in files_name[2]:
        img_path.append(directory+file_name)
img_path = img_path[1800:6000]
# for i in range(99):
#   if i<10:
#     root_directory = root+'0'+str(i)
#   else:
#     root_directory = root+str(i)
#   for root, dirs, files in os.walk(root_directory):
#     for files_name in files:
#       img_path.append(root+root_directory+files_name)


root_test = os.getcwd()
root_test += '\\wiki_crop\\00'
img_path_test = []
for root_test, dirs, files in os.walk(root_test):
    for files_name in files:
        img_path_test.append(root_test+'\\'+files_name)


'''
# * 第二步：加载全部图片为tensor（张量）
# todo 使用其他加载函数 Image.load()
'''
def decode_img(path):
    img_raw = tf.read_file(path)
    img_tensor = tf.io.decode_image(img_raw, channels=3)
    img_resize = tf.image.resize_images(img_tensor, [192, 192])
    # print(repr(img_raw)[:100]+"...")
    img_resize = tf.cast(img_resize, tf.float32)
    img_resize /= 255.0
    return img_resize

img_tensors = []
# for i in img_path:
#     img_tensors.append(decode_img(i))

img_tensors_test = []
for i in img_path_test:
    img_tensors_test.append(decode_img(i))


'''
# * 附加步骤，看一下图片
# * 开启show()才能看到
'''
import matplotlib.pyplot as plt

# plt.figure(figsize=(8,8))
# for n,image in enumerate(img_dataset.take(4)):
#   plt.subplot(2,2,n+1)
#   plt.imshow(image)
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   plt.xlabel(img_path[n])
#   plt.show()


''' 
# * 第三步：加载图片对应的年龄（来自于mat文件）
'''
import mat
# 组合img_dataset和mat
# 1.根据img_path重新排mat中age顺序
def save_to_record_file(file_path, age):
  # img_raw = tf.read_file(file_path)
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

for age in mat.age:
  if(age.get('filename') == '7721992_1947-10-06_1964.jpg'):
    print('debug!')
  
filename = 'pic_age_dataset_test.tfrecord'
writer = tf.python_io.TFRecordWriter(filename)
age_dataset=[]
img_dataset=[]
i=0
for file_path in img_path:
    print(i)
    file_path_sliced = file_path[51:]
    for age in mat.age:
        if age.get('filename') == file_path_sliced:
          # 排除负数数据
          if age.get('age') < 0:
            break
          else:
            age_dataset.append(int(age.get('age')))
            img_dataset.append(decode_img(file_path))
            # 在这里存成tfrecord file
            save_to_record_file(file_path, int(age.get('age')))
            break
    i +=1 

# age_dataset_test = []
# for file_path in img_path_test:
#     file_path_sliced = file_path[51:]
#     for age in mat.age_test:
#         if age.get('filename') == file_path_sliced:
#             age_dataset_test.append(int(age.get('age')))

# 检查一下数据顺序是否正确
# for age in mat.age:
#     if age.get('filename') == '10110600_1985-09-17_2012.jpg':
#         print(age['age'])


'''
# * 第四步：数据转为dataset,并取前n张（多了爆内存死机）
# todo 是否可以不转为dataset？直接训练
'''
img_dataset = img_dataset[:3000]
print("图片数:",len(img_dataset))
img_dataset = tf.data.Dataset.from_tensor_slices(img_dataset)
age_dataset = age_dataset[:3000]
print("标签数:",len(age_dataset))
age_dataset = tf.data.Dataset.from_tensor_slices(age_dataset)
image_label_ds = tf.data.Dataset.zip((img_dataset, age_dataset))



# img_tensors_test = img_tensors_test[:100]
# img_dataset_test = tf.data.Dataset.from_tensor_slices(img_tensors_test)
# age_dataset_test = age_dataset_test[:100]
# age_dataset_test = tf.data.Dataset.from_tensor_slices(age_dataset_test)
# image_label_ds_test = tf.data.Dataset.zip((img_dataset_test, age_dataset_test))
