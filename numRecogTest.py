import cv2
import tensorflow as tf
import numpy as np
from numData import x_test, y_test
import matplotlib.pyplot as plt

#during git push errors use 'git reset --soft HEAD~'
model = tf.keras.models.load_model('numMod.keras')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

# 10 means '+'
correct=[0,1,1,2,3,3,4,5,6,7,7,8,9]

for i in range(len(correct)):
    image= f"test_cases/test{i+1}.png"
    print(f"test{i+1}")
    img = cv2.imread(image)[:,:,0]
    img = np.invert(np.array([img]))
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)

    # plt.imshow(img[0], cmap="grey")
    # plt.title(f"Test {i+1}")
    # plt.show()

    prediction = np.argmax(model.predict(img))

    if prediction==correct[i]:
        print(f"This digit is correctly predicted as {prediction}")
    else:
        print(f"This digit was predicted as {prediction} and not {correct[i]}")
        all_true = False
        #work on fixing the little security breach here
    

