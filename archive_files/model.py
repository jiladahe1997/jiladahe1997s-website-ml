from main_eager import image_label_ds as ds
# from main_eager import image_label_ds_test as ds_test

import tensorflow as tf

'''
  # * 第一步：数据shuffle（打乱顺序）+ repeat + batch
  # ? repeat和batch的作用？
'''
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=600))
ds = ds.batch(20)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ds_test = ds_test.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=600))
# ds_test = ds_test.batch(18)
# ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

'''
  # * 第二步：数据[0-1]转换到[-1,1]
  # ? why？ 为什么要转换？教程说是mobilenet要求的，但是这里vgg19也要？
'''
def change_range(image,label):
  return image/255.0, label

ds = ds.map(change_range)
image_batch, label_batch = next(iter(ds))
# ds = tf.keras.applications.vgg19.preprocess_input(image_batch)
vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(192, 192, 3), weights='imagenet')

def loss_func(predict_y, desired_y):
  def process(y):
    for index, p in enumerate(predict_y):
      predict_y[index] = p*index
    return 
  predict_y = predict_y.map(process)
  return tf.reduce_mean(tf.square(predict_y - desired_y))
'''
  # * 第三步：建立sequential序列模型
  # ? 这里的输出层是Dense?
  # ? loss和metrics的选择
'''
# feature_shape = vgg19(image_batch)
model = tf.keras.Sequential([
    vgg19,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Softmax()
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='mean_absolute_error',
              metrics=["mean_absolute_error"])
model.summary()


'''
  # * 第四步：开始训练
  # todo 保存模型，修改steps_per_epoch
  # todo 增加可视化
'''
import pandas as pd
import matplotlib.pyplot as plt

history = model.fit(ds, epochs=20, steps_per_epoch=100)
model.save('my_model.h5')
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.ylim([0,100])
  plt.legend()
  
  plt.show()


plot_history(history)

a=0

'''
  # * 第五步：作出预测
'''
# import numpy as np
# prediction = model.predict_generator(ds_test, steps=3)
# evaluate = model.evaluate_generator(ds_test, steps=3)
# np.argmax(prediction[0])

a=[]
