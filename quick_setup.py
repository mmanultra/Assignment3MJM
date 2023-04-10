import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# load the data
test_data = np.load('test_data.npy')

# load the saved autoencoder
autoencoder = keras.models.load_model('autoencoder.h5')
# randomly select one image from the test data
idx = np.random.randint(0, len(test_data))
image = test_data[idx]

# resize the image to 56x56 using cv2
true_56 = cv2.resize(image, (56, 56), interpolation=cv2.INTER_CUBIC)

# preprocess the image for the autoencoder
image_28 = np.expand_dims(image, axis=0)
image_28 = np.expand_dims(image_28, axis=-1)

# generate the autoencoder's 28x28 and 56x56 predictions
predicted_28 = autoencoder.predict(image_28)
predicted_56 = cv2.resize(predicted_28[0].squeeze(), (56, 56), interpolation=cv2.INTER_CUBIC)

# display the images side by side
fig, ax = plt.subplots(1,4, figsize=(12,3))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original 28x28')
ax[0].set_aspect('equal')

ax[1].imshow(true_56, cmap='gray')
ax[1].set_title('True 56x56')
ax[1].set_aspect('equal')

ax[2].imshow(predicted_28.squeeze(), cmap='gray')
ax[2].set_title('Autoencoder 28x28')
ax[2].set_aspect('equal')

ax[3].imshow(predicted_56, cmap='gray')
ax[3].set_title('Autoencoder 56x56')
ax[3].set_aspect('equal')

plt.tight_layout()
plt.show()