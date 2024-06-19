import cv2
import tensorflow as tf
import numpy as np
from numData import x_test, y_test


model = tf.keras.models.load_model('numMod.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

img = "numbers_test/6.png"
img = cv2.imread(img)[:,:,0]
img = cv2.resize(img, (28,28))
img = np.invert(np.array([img]))
prediction = model.predict(img)
print(f"This digit is probably a {np.argmax(prediction)}")