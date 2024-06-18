#!/usr/bin/env python
# coding: utf-8

# ## 1. Importing libraries and Dataset Configuration

# In[1]:


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
import psycopg2
from dotenv import load_dotenv
import os


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('gpu ', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)


# ## 2. Database Connection and Retrieve

# In[3]:


# Load environment variables from .env file
load_dotenv()

# Verify environment variables are loaded correctly
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Print to check the loaded environment variables (for debugging purposes)
print(db_params)

# Connect to PostgreSQL database
try:
    conn = psycopg2.connect(**db_params)
    print("Connected to the database")
except Exception as e:
    print(f"Unable to connect to the database: {e}")


# In[4]:


query = "SELECT sentences, label_task FROM tbl1;"
df = pd.read_sql_query(query, conn)
df


# In[5]:


conn.close()
print("Connection closed")


# In[8]:


# Filter labels with at least 45 sentences
label_counts = df['label_task'].value_counts()
valid_labels = label_counts[label_counts >= 40].index

filtered_df = df[df['label_task'].isin(valid_labels)]
num_classes = filtered_df['label_task'].nunique()


# In[9]:


label_counts = filtered_df['label_task'].value_counts()
print(label_counts)


# In[7]:


import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('text_classify.h5')

# Load the saved tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)


# In[16]:


# Preprocess and tokenize the new data
new_sentences = filtered_df['sentences'].tolist()
new_labels = filtered_df['label_task'].tolist()

new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded = pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')

new_labels_encoded = label_encoder.transform(new_labels)
new_labels_one_hot = tf.keras.utils.to_categorical(new_labels_encoded, num_classes=len(label_encoder.classes_))


# In[17]:


# Fine-tune the loaded model with the new data
history = model.fit(new_padded, 
                    new_labels_one_hot, 
                    epochs=5,  # A few epochs for fine-tuning
                    validation_split=0.2)


# In[ ]:


# Save the updated model
model.save('text_classify_feedbacked.h5')

