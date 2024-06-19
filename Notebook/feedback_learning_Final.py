from flask import Flask, jsonify
import os
import tensorflow as tf
import pandas as pd
import pickle
import psycopg2
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def run_model():
    load_dotenv()

    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to the database")
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
        return jsonify({"error": str(e)}), 500

    query = "SELECT sentences, label_task FROM tbl1;"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print("Connection closed")

    label_counts = df['label_task'].value_counts()
    valid_labels = label_counts[label_counts >= 3].index
    filtered_df = df[df['label_task'].isin(valid_labels)]
    num_classes = filtered_df['label_task'].nunique()

    model = tf.keras.models.load_model('Notebook/text_classify.h5')

    with open('Notebook/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('Notebook/label_encoder.pkl', 'rb') as handle:
        label_encoder = pickle.load(handle)

    new_sentences = filtered_df['sentences'].tolist()
    new_labels = filtered_df['label_task'].tolist()
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    new_padded = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')
    new_labels_encoded = label_encoder.transform(new_labels)
    new_labels_one_hot = tf.keras.utils.to_categorical(new_labels_encoded, num_classes=len(label_encoder.classes_))

    history = model.fit(new_padded, new_labels_one_hot, epochs=5, validation_split=0.2)
    model.save('Notebook/text_classify_feedbacked.h5')

    return jsonify({"message": "Model training complete"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
