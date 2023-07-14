import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# visualisation
# plt.imshow(x_train[0])
# plt.show()
x_train = x_train / 255.0
x_test = x_test / 255.0
physical_gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.set_logical_device_configuration(physical_gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
logical_gpu = tf.config.list_logical_devices('GPU')[0]
with tf.device(logical_gpu):
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                 tf.keras.layers.Dense(128, activation="relu"),
                                 tf.keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model = tf.keras.models.load_model('model0.8891')
model.fit(x_train, y_train, epochs=10)
print('Проверка на тестовом наборе:')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss, ', Test accuracy:', test_acc)
print('Проверка на конкретном примере:')
predictions = model.predict(x_test)
prediction = np.argmax(predictions[0])
print('Ожидаемое значение:', y_test[0], ', значение предсказанное моделью:', prediction)
Model_name = 'model' + str((test_acc // 0.0001) * 0.0001)
model.save(Model_name)
