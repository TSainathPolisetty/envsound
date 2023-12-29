import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json

# Load and preprocess image data
def load_data(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = load_img(os.path.join(directory, filename), color_mode='grayscale', target_size=(256, 256))
            img_array = img_to_array(img)
            images.append(img_array)
            
            # Extract label from filename
            label = filename.split('_')[2]
            labels.append(label)
            # print(labels)

    return np.array(images), np.array(labels)

# Encode labels into integers
def encode_labels(labels):
    unique_labels = np.unique(labels)
    label_to_int = dict((label, i) for i, label in enumerate(unique_labels))
    int_labels = np.array([label_to_int[label] for label in labels])
    # Print the mapping of label to integer
    print("Label to Integer Mapping:")
    for label, integer in label_to_int.items():
        print(f"'{label}': {integer}")
    return int_labels, len(unique_labels)

# Directory containing processed data
data_dir = '/home/tulasi/sig/cnndata' #please replace with the cnndata file path once you extract

# Load and preprocess data
images, labels = load_data(data_dir)
images = images / 255.0  # Normalize the images

# Encode labels and convert to categorical
encoded_labels, num_classes = encode_labels(labels)
categorical_labels = to_categorical(encoded_labels)

# Split the dataset
X_train_images, X_test_images, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.33, random_state=42)

# Define the CNN model for PWVD images
pwvd_model = Sequential()
pwvd_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
pwvd_model.add(MaxPooling2D((2, 2)))
pwvd_model.add(Conv2D(64, (3, 3), activation='relu'))
pwvd_model.add(MaxPooling2D((2, 2)))
pwvd_model.add(Flatten())

# Load entropy features
with open('/home/tulasi/sig/cnndata/entropy_features.json', 'r') as f:
    entropy_features = json.load(f)

# Prepare additional features and ensure they are aligned with the images
additional_features = []
for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        unique_id = filename.split('_')[0]
        features = entropy_features.get(unique_id, [])
        additional_features.append(features)

additional_features = np.array(additional_features)

# Split additional features
X_train_features, X_test_features = train_test_split(additional_features, test_size=0.33, random_state=42)

# Dense Model for additional features
input_features = Input(shape=(additional_features.shape[1],))
y = Dense(64, activation='relu')(input_features)
feature_model = Model(inputs=input_features, outputs=y)

# Combine Models
combined_input = concatenate([pwvd_model.output, feature_model.output])
z = Dense(64, activation='relu')(combined_input)
z = Dense(num_classes, activation='softmax')(z)  # Use softmax for multi-class classification

combined_model = Model(inputs=[pwvd_model.input, feature_model.input], outputs=z)
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model and capture the history
history = combined_model.fit([X_train_images, X_train_features], y_train, epochs=100, validation_data=([X_test_images, X_test_features], y_test), batch_size=32)

# Evaluate the model and plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Confusion matrix
y_pred = combined_model.predict([X_test_images, X_test_features])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred_classes))

# Print model summary and save the model
combined_model.summary()
combined_model.save('sound_classification_combined_model.h5')

print("Combined model training and evaluation completed.")
