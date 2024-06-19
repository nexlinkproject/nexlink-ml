import os
from google.cloud import bigquery, pubsub_v1

PROJECT_ID = os.getenv('GCP_PROJECT')
DATASET_ID = 'nexlink_dataset'
TABLE_ID = 'feedback_data'
PROCESS_TABLE_ID = 'processed_count'
PUBSUB_TOPIC = 'projects/nexlink-prod/topics/retrain-model'

def check_new_data(event, context):
    client = bigquery.Client()

    # Get current total row count
    query_total_count = f"""
    SELECT COUNT(*) AS total_rows
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    """
    result = client.query(query_total_count).result()
    total_row_count = [row.total_rows for row in result][0]

    # Get last processed row count
    query_last_count = f"""
    SELECT last_count
    FROM `{PROJECT_ID}.{DATASET_ID}.{PROCESS_TABLE_ID}`
    LIMIT 1
    """
    result = client.query(query_last_count).result()
    last_processed_count = [row.last_count for row in result][0]

    # Check if 100 new rows have been added
    if total_row_count - last_processed_count >= 100:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
        publisher.publish(topic_path, b'Triggering model retrain')

        # Update last processed count
        update_query = f"""
        UPDATE `{PROJECT_ID}.{DATASET_ID}.{PROCESS_TABLE_ID}`
        SET last_count = {total_row_count}
        """
        client.query(update_query).result()