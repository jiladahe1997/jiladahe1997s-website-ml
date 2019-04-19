

 
import tensorflow as tf



raw_image_dataset = tf.data.TFRecordDataset('pic_age_dataset_test.tfrecord')

image_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
    'raw': tf.FixedLenFeature([], tf.string),
    # 'file_path': tf.FixedLenFeature([], tf.string)
}
def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  single_example = tf.parse_single_example(example_proto, image_feature_description)
  raw  = single_example['raw']
  age = single_example['label']
  image_tensor = tf.io.decode_jpeg(raw, channels=3)
  image_resize_tensor = tf.cast(tf.image.resize_images(image_tensor, [192, 192]), tf.float32)
  print(image_resize_tensor,age)
  return [image_resize_tensor, age]
#   return tf.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)

with tf.Session() as sess:
#   sess.run(parsed_image_dataset)
  print(parsed_image_dataset)

image_tensors = []
age_tensors = []

# import matplotlib.pyplot as plt
# # 读取数据
# for each_image_feature in parsed_image_dataset:
#   raw = each_image_feature['raw']
#   image_tensor = tf.io.decode_image(raw, channels=3)
#   # plt.imshow(image_tensor.numpy())
#   # plt.show()
#   image_tensor = tf.cast(tf.image.resize_images(image_tensor, [192,192]),tf.float32)
#   # plt.imshow(image_tensor.numpy())
#   # plt.show()
#   image_tensors.append(image_tensor/255.0)
#   age_tensors.append(each_image_feature['label'])
#   if len(image_tensors) == 10000:
#     break

# image_tensors = tf.Variable(image_tensors, tf.float32)
# age_tensors = tf.Variable(age_tensors, tf.float32)
# age_tensors = tf.convert_to_tensor(age_tensors)









# print("图片数:",len(image_tensors))
# img_dataset = tf.data.Dataset.from_tensor_slices(image_tensors)
# print("标签数:",len(age_tensors))
# age_dataset = tf.data.Dataset.from_tensor_slices(age_tensors)
# image_label_ds = tf.data.Dataset.zip((img_dataset, age_dataset))
# ds = image_label_ds.apply(
#   tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
# ds = ds.batch(20)
# ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds = parsed_image_dataset.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=600))
ds = ds.batch(20)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)








vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(192, 192, 3), weights='imagenet')
# model = tf.keras.Sequential([
#     vgg19,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512),
#     tf.keras.layers.Dense(1),
# ])
model = tf.keras.models.load_model('my_model.h5')
# image_tensors=tf.keras.applications.vgg19.preprocess_input(image_tensors)
# age_tensors==tf.keras.applications.vgg19.preprocess_input(age_tensors)
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])
model.summary()
history = model.evaluate(ds , steps=30)
# model.save('my_model1.h5')





# 500/500 [==============================] - 292s 585ms/step - loss: 1435.8452 - mean_absolute_error: 1435.8459
# Epoch 2/20
# 500/500 [==============================] - 267s 535ms/step - loss: 17.3193 - mean_absolute_error: 17.3193
# Epoch 3/20
# 500/500 [==============================] - 267s 534ms/step - loss: 17.4023 - mean_absolute_error: 17.4023
# Epoch 4/20
# 500/500 [==============================] - 268s 535ms/step - loss: 16.8086 - mean_absolute_error: 16.8086
# Epoch 5/20
# 500/500 [==============================] - 268s 536ms/step - loss: 16.5949 - mean_absolute_error: 16.5949
# Epoch 6/20
# 500/500 [==============================] - 265s 530ms/step - loss: 16.3916 - mean_absolute_error: 16.3916
# Epoch 7/20
# 500/500 [==============================] - 266s 532ms/step - loss: 14.4644 - mean_absolute_error: 14.4644
# Epoch 8/20
# 500/500 [==============================] - 270s 540ms/step - loss: 13.6874 - mean_absolute_error: 13.6874
# Epoch 9/20
# 500/500 [==============================] - 270s 541ms/step - loss: 13.4170 - mean_absolute_error: 13.4170
# Epoch 10/20
# 500/500 [==============================] - 266s 533ms/step - loss: 13.3024 - mean_absolute_error: 13.3024
# Epoch 11/20
# 500/500 [==============================] - 267s 535ms/step - loss: 13.2153 - mean_absolute_error: 13.2153
# Epoch 12/20
# 500/500 [==============================] - 267s 535ms/step - loss: 13.4479 - mean_absolute_error: 13.4479
# Epoch 13/20
# 500/500 [==============================] - 266s 532ms/step - loss: 13.1466 - mean_absolute_error: 13.1466
# Epoch 14/20
# 500/500 [==============================] - 265s 530ms/step - loss: 13.2886 - mean_absolute_error: 13.2886
# Epoch 15/20
# 500/500 [==============================] - 265s 530ms/step - loss: 13.2253 - mean_absolute_error: 13.2254
# Epoch 16/20
# 500/500 [==============================] - 269s 538ms/step - loss: 12.9313 - mean_absolute_error: 12.9313
# Epoch 17/20
# 500/500 [==============================] - 270s 540ms/step - loss: 12.8665 - mean_absolute_error: 12.8665
# Epoch 18/20
# 500/500 [==============================] - 267s 535ms/step - loss: 13.2932 - mean_absolute_error: 13.2932
# Epoch 19/20
# 500/500 [==============================] - 267s 534ms/step - loss: 12.4842 - mean_absolute_error: 12.4842
# Epoch 20/20
# 500/500 [==============================] - 267s 534ms/step - loss: 12.5006 - mean_absolute_error: 12.5006



a=0