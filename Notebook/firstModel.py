#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ## 1. Importing libraries and Dataset Configuration

# In[28]:


import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import re
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('gpu ', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)


# #### Dataset Pipelining

# In[10]:


df = pd.read_csv('dataset5.csv')
df.head()


# #### Dataset Exploration and Cleaning

# In[11]:


df.info()


# In[12]:


print("Jumlah duplikasi: ",df.duplicated().sum())
df.describe()


# In[13]:


df = df.drop_duplicates()
print("Jumlah duplikasi: ",df.duplicated().sum())


# In[14]:


# Drop rows where 'Sentences' column has NaN values
df = df.dropna(subset=['Sentences'])

# Calculate the counts of each label
label_counts = df['Label_Task'].value_counts()
print(label_counts.head(26))


# ## 2. Preprocessing Dataset

# #### Splitting Data

# In[15]:


# Filter labels with at least 45 sentences
valid_labels = label_counts[label_counts >= 80].index

filtered_df = df[df['Label_Task'].isin(valid_labels)]
num_classes = filtered_df['Label_Task'].nunique()

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels to integers
filtered_df.loc[:, 'Label_Task'] = label_encoder.fit_transform(filtered_df['Label_Task'])

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(filtered_df, train_size=0.8,stratify=filtered_df['Label_Task'], random_state=42, shuffle=True)

# Extract sentences and labels for training
train_sentences = train_df['Sentences'].tolist()
train_labels = train_df['Label_Task'].tolist()

# Extract sentences and labels for validation
val_sentences = val_df['Sentences'].tolist()
val_labels = val_df['Label_Task'].tolist()

# Check the results
print(f"Number of training examples: {len(train_sentences)}")
print(f"Number of validation examples: {len(val_sentences)}")

# print('First 5 train sentences:\n', train_sentences[:5])
# print('First 5 train labels:\n', train_labels[:5])

# print('First 5 val sentences:\n', val_sentences[:5])
# print('First 5 val labels:\n', val_labels[:5])


# In[16]:


print('val labels:',num_classes)


# #### Lowercase Augmentation

# In[17]:


def lowercase_augmentation(sentences):
    augmented_sentences = []
    for sentence in sentences:
        lowercase_sentence = sentence.lower()
        augmented_sentences.append(lowercase_sentence)
    return augmented_sentences


# #### Tokenizer

# In[18]:


vocab_size = 100000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
# Melakukan data augmentasi pada data latih dengan mengubah semua kalimat menjadi huruf kecil
train_sentences = lowercase_augmentation(train_sentences)
val_sentences = lowercase_augmentation(val_sentences)

# Tokenize the training sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# Create sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

# Convert labels to one-hot encoding if using categorical_crossentropy
# Ensure labels are converted to one-hot encoding only once
if len(train_labels.shape) == 1:
    train_labels = to_categorical(train_labels, num_classes)
if len(val_labels.shape) == 1:
    val_labels = to_categorical(val_labels, num_classes)

# Check the shape of the padded sequences and labels
print(f"Training data shape: {train_padded.shape}")
print(f"Validation data shape: {val_padded.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Validation labels shape: {val_labels.shape}")


# ## 3. Model Building and Training

# In[19]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


# In[20]:


model.summary()


# #### Model Training

# In[21]:


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# Define the ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',  # Path where the model weights will be saved
    monitor='val_loss',        # Metric to monitor
    save_best_only=True,       # Save the best model only
    save_weights_only=False,   # Save the entire model (including architecture)
    mode='min',                # Mode for monitoring ('min' for loss, 'max' for accuracy)
    verbose=1                  # Verbosity mode, 1 for progress updates
)

# Define other callbacks (optional)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

callback_list = [
    reduce_lr_callback,
    early_stopping_callback
]


# In[22]:


batch_size = 64
num_epochs = 40
history = model.fit(train_padded, 
                    train_labels, 
                    epochs=num_epochs, 
                    # batch_size=batch_size, 
                    validation_data=(val_padded, val_labels), 
                    callbacks=callback_list,
                    verbose=1)


# #### Model Predict

# In[27]:


# Example input sentences
new_sentences = ["Melakukan analisis pasar untuk mengidentifikasi target pelanggan potensial website, serta preferensi mereka dalam berbelanja online.", 
                 "Mengamankan akses ke platform cloud dan aplikasi web dengan menerapkan autentikasi dan otorisasi yang sesuai."]

new_sentences = ["buat front end", 
                 "analisis keperluan user"]
# Tokenize the sentences
sequences = tokenizer.texts_to_sequences(new_sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
# Make predictions
predictions = model.predict(padded_sequences)
# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted class indices:", predicted_classes)
# Convert numeric labels back to original string labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)
print("Predicted class labels:", predicted_labels)


# #### Model, Tokenizer, and Label Encoder Save

# In[26]:


model.save("text_classify.h5")


# In[29]:


# Save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ## 4. Evaluation

# In[25]:


import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Predict on validation data
predictions = model.predict(val_padded)
y_true = np.argmax(val_labels, axis=1)
y_pred = np.argmax(predictions, axis=1)

# Ensure the number of classes match
unique_labels = np.unique(y_true)
target_names = [label_encoder.inverse_transform([i])[0] for i in unique_labels]

# Generate classification report
report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

# Print classification report
print(report)

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

