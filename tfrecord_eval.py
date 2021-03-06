import tensorflow as tf

raw_image_dataset = tf.data.TFRecordDataset('pic_age_dataset_test.tfrecord')

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





model = tf.keras.models.load_model('my_model.h5')
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])
model.summary()
model.evaluate(ds , steps=30)

