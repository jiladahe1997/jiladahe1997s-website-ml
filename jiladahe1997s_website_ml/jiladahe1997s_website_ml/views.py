from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import os

'''
协议：http
请求参数：来自前端的base64编码的图片
返回参数：模型预测的年龄
'''

# 注：每次比如重新load一遍model，如果你想节约内存只在加载views.py的时候加载一次，否则报某个不知名的错，作者谷歌百度了3个小时都没有解决。
# model = tf.keras.models.load_model(os.path.abspath('./my_model.h5'))

# 由于前端的请求来自于非同源url，因此需要csrf跨域共享
@csrf_exempt
def age_predict(request):
  import tensorflow as tf
  import os
  import numpy as np
  import base64
  import json
  model = tf.keras.models.load_model(os.path.abspath('./my_model.h5'))
  model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])

  # * 处理前端的base64字符串，去掉头部，
  # * 并且转换为urlsafe形式，这是TensorFlow的要求，详情见  https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/decode_base64?hl=en    
  img_base64_header, img_base64 = json.loads(request.body)['base64_img'].split(",",1)
  raw = base64.decodestring(img_base64.encode('utf-8'))
  img_base64_websafe = base64.urlsafe_b64encode(raw)

  # * base转为tensor，然后输入模型
  img_raw = tf.decode_base64(img_base64_websafe)
  image_tensor = tf.cast(tf.image.resize_images(tf.io.decode_jpeg(img_raw, channels=3), [192,192]),tf.float32)
  image = tf.expand_dims(image_tensor, axis=0)
  
  age = model.predict(image, steps=1)
  return HttpResponse(age) 