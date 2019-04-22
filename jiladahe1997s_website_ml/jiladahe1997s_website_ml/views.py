from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import os


model = tf.keras.models.load_model(os.path.abspath('./my_model.h5'))
model.compile(optimizer=tf.train.AdamOptimizer(), 
            loss="mean_absolute_error",
            metrics=["mean_absolute_error"])
@csrf_exempt
def age_predict(request):
  # todo 走一遍机器学习
  import tensorflow as tf
  import os
  import numpy as np
  import base64
  import json
  print(os.path.abspath('../my_model.h5'))
  # * 模拟原生的二进制数据
  # img_raw = open(os.path.abspath('./QQ截图20190420203720.png'), 'rb').read()
  img_base64_header, img_base64 = json.loads(request.body)['base64_img'].split(",",1)
  raw = base64.decodestring(img_base64.encode('utf-8'))
  img_base64_websafe = base64.urlsafe_b64encode(raw)
  img_raw = tf.decode_base64(img_base64_websafe)
  # * 模拟结束
  image_tensor = tf.cast(tf.image.resize_images(tf.io.decode_jpeg(img_raw, channels=3), [192,192]),tf.float32)
  image = tf.expand_dims(image_tensor, axis=0)
  age = model.predict(image, steps=1) 
  return HttpResponse(age) 