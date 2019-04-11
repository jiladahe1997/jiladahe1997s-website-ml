import tensorflow as tf
tf.enable_eager_execution()

raw_image_dataset = tf.data.TFRecordDataset('pic_age_dataset_test_9.tfrecord')

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
  return tf.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

image_tensors = []
age_tensors = []

import matplotlib.pyplot as plt
# 读取数据
for each_image_feature in parsed_image_dataset:
  raw = each_image_feature['raw']
  image_tensor = tf.io.decode_image(raw, channels=3)
  # plt.imshow(image_tensor.numpy())
  # plt.show()
  image_tensor = tf.cast(tf.image.resize_images(image_tensor, [192,192]),tf.float32)
  # plt.imshow(image_tensor.numpy())
  # plt.show()
  image_tensors.append(image_tensor/255)
  age_tensors.append(each_image_feature['label'])

# image_tensors = tf.Variable(image_tensors, tf.float32)
# age_tensors = tf.Variable(age_tensors, tf.float32)
# age_tensors = tf.convert_to_tensor(age_tensors)

image_tensors = image_tensors[:600]
age_tensors = age_tensors[:600]







print("图片数:",len(image_tensors))
img_dataset = tf.data.Dataset.from_tensor_slices(image_tensors)
print("标签数:",len(age_tensors))
age_dataset = tf.data.Dataset.from_tensor_slices(age_tensors)
image_label_ds = tf.data.Dataset.zip((img_dataset, age_dataset))
ds = image_label_ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=600))
ds = ds.batch(20)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)









vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(192, 192, 3), weights='imagenet')
# model = tf.keras.Sequential([
#     vgg19,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(1),
# ])
model = tf.keras.models.load_model('my_model1.h5')

# image_tensors=tf.keras.applications.vgg19.preprocess_input(image_tensors)
# age_tensors==tf.keras.applications.vgg19.preprocess_input(age_tensors)
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])
model.summary()
history = model.fit(ds , epochs=20, steps_per_epoch=30)
model.save('my_model1.h5')




# 30/30 [==============================] - 18s 589ms/step - loss: 14.1461 - mean_absolute_error: 14.1461
# 30/30 [==============================] - 18s 591ms/step - loss: 17.6492 - mean_absolute_error: 17.6492
# 30/30 [==============================] - 18s 596ms/step - loss: 12.8466 - mean_absolute_error: 12.8466
# 30/30 [==============================] - 18s 591ms/step - loss: 13.8928 - mean_absolute_error: 13.8928
# 30/30 [==============================] - 18s 591ms/step - loss: 13.9583 - mean_absolute_error: 13.9583
# 30/30 [==============================] - 18s 588ms/step - loss: 14.2920 - mean_absolute_error: 14.2920
# 30/30 [==============================] - 18s 590ms/step - loss: 13.2410 - mean_absolute_error: 13.2410
# 30/30 [==============================] - 18s 590ms/step - loss: 13.4772 - mean_absolute_error: 13.4772
# 30/30 [==============================] - 18s 591ms/step - loss: 13.3961 - mean_absolute_error: 13.3961
# 30/30 [==============================] - 18s 591ms/step - loss: 13.4301 - mean_absolute_error: 13.4301




a=0