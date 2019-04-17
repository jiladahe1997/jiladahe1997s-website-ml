from django.http import HttpResponse


def age_predict(request):
  # todo 走一遍机器学习
  import tensorflow as tf
  import os
  import numpy as np
  print(os.path.abspath('../my_model.h5'))
  # * 模拟原生的二进制数据
  img_raw = open(os.path.abspath('../QQ截图20190416205544.png'), 'rb').read()
  # * 模拟结束
  image_tensor = tf.cast(tf.image.resize_images(tf.io.decode_jpeg(img_raw, channels=3), [192,192]),tf.float32)
  print(image_tensor)
  model = tf.keras.models.load_model(os.path.abspath('../my_model.h5'))
  model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])
  image = tf.expand_dims(image_tensor, axis=0)
  age = model.predict(image, steps=1) 
  return HttpResponse(age) 