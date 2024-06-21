import cv2
import tensorflow as tf
import numpy as np
from numData import x_test, y_test
from mnistUpdate import update_mnist
from numRecogTrain import train


model = tf.keras.models.load_model('numMod.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

# 10 means '+'
correct=[9,0,5,6,10,10,10,10,10,10]

for i in range(len(correct)):
    image= f"test_train_data/test{i}.png"
    print(f"test{i}")
    img = cv2.imread(image)[:,:,0]
    img = cv2.resize(img, (28,28))
    img = np.invert(np.array([img]))
    prediction = np.argmax(model.predict(img))

    if prediction==correct[i]:
        print(f"This digit is correctly predicted as {prediction}")
    else:
        print(f"This digit was predicted as {prediction} and not {correct[i]}")
        update_mnist(img, correct[i])
        all_true = False
        #work on fixing the little security breach here

train()