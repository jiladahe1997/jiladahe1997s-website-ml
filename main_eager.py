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
root += '\\wiki_crop\\00'
img_path = []
for root, dirs, files in os.walk(root):
    for files_name in files:
        img_path.append(root+'\\'+files_name)

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
    img_resize /= 255.0
    return img_resize

img_tensors = []
for i in img_path:
    img_tensors.append(decode_img(i))

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
age_dataset=[]
for file_path in img_path:
    file_path_sliced = file_path[51:]
    for age in mat.age:
        if age.get('filename') == file_path_sliced:
            age_dataset.append(int(age.get('age')))

age_dataset_test = []
for file_path in img_path_test:
    file_path_sliced = file_path[51:]
    for age in mat.age_test:
        if age.get('filename') == file_path_sliced:
            age_dataset_test.append(int(age.get('age')))

# 检查一下数据顺序是否正确
# for age in mat.age:
#     if age.get('filename') == '10110600_1985-09-17_2012.jpg':
#         print(age['age'])


'''
# * 第四步：数据转为dataset,并取前n张（多了爆内存死机）
# todo 是否可以不转为dataset？直接训练
'''
img_tensors = img_tensors[:600]
img_dataset = tf.data.Dataset.from_tensor_slices(img_tensors)
age_dataset = age_dataset[:600]
age_dataset = tf.data.Dataset.from_tensor_slices(age_dataset)
image_label_ds = tf.data.Dataset.zip((img_dataset, age_dataset))

img_tensors_test = img_tensors_test[:100]
img_dataset_test = tf.data.Dataset.from_tensor_slices(img_tensors_test)
age_dataset_test = age_dataset_test[:100]
age_dataset_test = tf.data.Dataset.from_tensor_slices(age_dataset_test)
image_label_ds_test = tf.data.Dataset.zip((img_dataset_test, age_dataset_test))
