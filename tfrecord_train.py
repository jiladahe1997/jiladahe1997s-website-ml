import tensorflow as tf

raw_image_dataset = tf.data.TFRecordDataset('pic_age_dataset.tfrecord')

def _parse_image_function(example_proto):
  image_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
    'raw': tf.FixedLenFeature([], tf.string),
  }
  single_example = tf.parse_single_example(example_proto, image_feature_description)
  raw  = single_example['raw']
  age = single_example['label']
  image_tensor = tf.io.decode_jpeg(raw, channels=3)
  image_resize_tensor = tf.cast(tf.image.resize_images(image_tensor, [192, 192]), tf.float32)
  print(image_resize_tensor,age)
  return [image_resize_tensor, age]



parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
ds = parsed_image_dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=600))
ds = ds.batch(20)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)




vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(192, 192, 3), weights='imagenet')
model = tf.keras.Sequential([
    vgg19,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])
model.summary()
history = model.fit(ds , epochs=20, steps_per_epoch=30)
model.save('my_model.h5')



