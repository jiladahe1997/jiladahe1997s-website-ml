import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path as path
import os

tf.enable_eager_execution()
print(tf.__version__, keras.__version__)
root = os.getcwd()
root += '\\wiki_crop\\00'
img_path = []
for root, dirs, files in os.walk(root):
    for files_name in files:
        img_path.append(root+'\\'+files_name)
    
# pro_path = path()
# image_path = pro_path/'wiki_crop'/'00'
# image_root = list(image_path.glob('**/*.*'))
# image_root = [str(i) for i in image_root]

# 读取图片
test_str = 'F:\\test.jpg'
def decode_img(path):
    img_raw = tf.read_file(test_str)
    img = tf.io.decode_image(img_raw, channels=3)
    img /= 255.0
    print(repr(img_raw)[:100]+"...")
    return img

a = []
# for i in img_path:
#     a.append(decode_img(i))


img_raw = tf.read_file(test_str)
print(repr(img_raw)[:100]+"...")

# image_generator = keras.preprocessinsg.image.ImageDataGenerator(rescale=1/255)
# image_data = image_generator.flow_from_directory('wiki_crop')
