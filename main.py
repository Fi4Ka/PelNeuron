import os
import random
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_images(folder_path, num_images):
    image_files = os.listdir(folder_path)
    random.shuffle(image_files)
    image_files = image_files[:num_images]

    images = []
    labels = []
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        images.append(image)

        age = int(file.split('_')[0])
        labels.append(age)

    return np.array(images), np.array(labels)

train_images, train_labels = load_images('fotos/', num_images=1000)
test_images, test_labels = load_images('fotos/', num_images=300)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_images, train_labels, epochs=10, batch_size=10)

loss = model.evaluate(test_images, test_labels)
print("Test data loss", loss)

predictions = model.predict(test_images)
rounded_predictions = np.round(predictions)

fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axs.flat):
    image = Image.fromarray(np.uint8(train_images[i] * 255))

    ax.imshow(image)
    ax.axis('off')

    real_age = test_labels[i]
    predicted_age = rounded_predictions[i][0]

    ax.set_title(f"Real: {real_age}\nPredicted: {predicted_age}")

plt.tight_layout()
plt.show()