from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
    
model.load_weights(model_save_path)

preNum = 1

for i in range(preNum):
    image_path = "9.png"
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    img_arr = 255 - img_arr
                
    img_arr = img_arr / 255.0
    print("img_arr:",img_arr.shape)
    x_predict = img_arr[tf.newaxis, ...]
    print("x_predict:",x_predict.shape)
    result = model.predict(x_predict)
    print(result)
    pred = tf.argmax(result[0])
    print(pred)
    tf.print(pred)
