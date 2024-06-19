from flask import Flask, jsonify, request
import os
import tensorflow as tf
import pandas as pd
import pickle
from google.cloud import bigquery, storage
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/', methods=['POST'])
def run_model():
    if not request.json or 'message' not in request.json:
        return jsonify({"error": "Invalid Pub/Sub message format"}), 400

    pubsub_message = request.json['message']['data']
    if pubsub_message != 'Triggering model retrain':
        return jsonify({"error": "Unexpected Pub/Sub message data"}), 400

    PROJECT_ID = os.getenv('GCP_PROJECT')
    DATASET_ID = 'nexlink_dataset'
    TABLE_ID = 'feedback_data'
    BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
    MODEL_FILENAME = 'text_classify_feedbacked.h5'
    TOKENIZER_FILENAME = 'tokenizer_feedbacked.pkl'
    LABEL_ENCODER_FILENAME = 'label_encoder_feedbacked.pkl'

    client = bigquery.Client()

    # Query data from BigQuery
    query = f"""
    SELECT sentences, label_task
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    df = client.query(query).to_dataframe()

    if df.empty:
        return jsonify({"message": "No data to train on"}), 204

    # Filter and process the data
    label_counts = df['label_task'].value_counts()
    valid_labels = label_counts[label_counts >= 3].index
    filtered_df = df[df['label_task'].isin(valid_labels)]
    num_classes = filtered_df['label_task'].nunique()

    # Load the model and preprocessing tools
    model = load_model('Notebook/text_classify.h5')

    with open('Notebook/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('Notebook/label_encoder.pkl', 'rb') as handle:
        label_encoder = pickle.load(handle)

    # Prepare data for training
    new_sentences = filtered_df['sentences'].tolist()
    new_labels = filtered_df['label_task'].tolist()
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    new_padded = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')
    new_labels_encoded = label_encoder.transform(new_labels)
    new_labels_one_hot = tf.keras.utils.to_categorical(new_labels_encoded, num_classes=len(label_encoder.classes_))

    # Train the model
    history = model.fit(new_padded, new_labels_one_hot, epochs=5, validation_split=0.2)
    
    # Save the retrained model locally
    model.save(f'Notebook/{MODEL_FILENAME}')
    with open(f'Notebook/{TOKENIZER_FILENAME}', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    with open(f'Notebook/{LABEL_ENCODER_FILENAME}', 'wb') as handle:
        pickle.dump(label_encoder, handle)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILENAME)
    blob.upload_from_filename(f'Notebook/{MODEL_FILENAME}')
    
    blob = bucket.blob(TOKENIZER_FILENAME)
    blob.upload_from_filename(f'Notebook/{TOKENIZER_FILENAME}')
    
    blob = bucket.blob(LABEL_ENCODER_FILENAME)
    blob.upload_from_filename(f'Notebook/{LABEL_ENCODER_FILENAME}')

    return jsonify({"message": "Model training and uploading complete"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
