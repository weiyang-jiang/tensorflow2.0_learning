import tensorflow as tf
tf.config.list_physical_devices('GPU')
print('GPU', tf.test.is_gpu_available())

a = tf.constant(2.0)
b = tf.constant(4.0)
print(a + b)