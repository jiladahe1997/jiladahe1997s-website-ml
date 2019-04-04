from main_eager import image_label_ds as ds
import tensorflow as tf

ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=600))
ds = ds.batch(32)
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def change_range(image,label):
  return 2*image-1, label

ds = ds.map(change_range)
image_batch, label_batch = next(iter(ds))
# ds = tf.keras.applications.vgg19.preprocess_input(image_batch)
vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(192, 192, 3), weights='imagenet')

# feature_shape = vgg19(image_batch)
model = tf.keras.Sequential([
    vgg19,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(100)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss="mean_absolute_error",
              metrics=["mean_absolute_error"])
model.summary()
model.fit(ds, epochs=3, steps_per_epoch=3)
a=0