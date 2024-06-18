#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from tensorflow.keras.models import load_model
import pickle


# In[4]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('gpu ', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)


# In[5]:


# Load the saved model
loaded_model = load_model("text_classify.h5")

# Load tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)


# In[19]:


# Define your max_length
max_length = 100  # Adjust this based on your preprocessing

def predict_task_labels(model, tokenizer, label_encoder, tasks):
    sequences = tokenizer.texts_to_sequences(tasks)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels

def get_task_durations(task):
    data = pd.read_csv('dataset5.csv')
    data_choice = data[data['Label_Task'] == task]
    if data_choice.empty:
        raise ValueError(f"Task '{task}' not found in the dataset.")
    duration = data_choice['Estimated_hours'].iloc[0]
    return duration

def assign_workers_to_tasks(task_labels, task_workers):
    worker_assignments = {}
    for idx, (task, worker) in enumerate(zip(task_labels, task_workers)):
        unique_task = f"{task}_{idx}"  # Ensure each task is unique
        duration = get_task_durations(task)
        worker_assignments[unique_task] = (worker, duration)
    return worker_assignments

def calculate_earliest_times(tasks, dependencies):
    earliest_start = {task: 0 for task in tasks}
    earliest_finish = {task: duration for task, (worker, duration) in tasks.items()}
    
    adj_list = defaultdict(list)
    in_degree = {task: 0 for task in tasks}
    
    for task, deps in dependencies.items():
        for dep in deps:
            adj_list[dep].append(task)
            in_degree[task] += 1
    
    topo_order = []
    zero_in_degree_queue = deque([task for task in tasks if in_degree[task] == 0])
    
    while zero_in_degree_queue:
        task = zero_in_degree_queue.popleft()
        topo_order.append(task)
        
        for neighbor in adj_list[task]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)
    
    for task in topo_order:
        for neighbor in adj_list[task]:
            earliest_start[neighbor] = max(earliest_start[neighbor], earliest_finish[task])
            earliest_finish[neighbor] = earliest_start[neighbor] + tasks[neighbor][1]
    
    return earliest_start, earliest_finish

def calculate_latest_times(tasks, dependencies, project_duration):
    latest_finish = {task: project_duration for task in tasks}
    latest_start = {task: project_duration - duration for task, (worker, duration) in tasks.items()}
    
    adj_list = defaultdict(list)
    for task, deps in dependencies.items():
        for dep in deps:
            adj_list[task].append(dep)
    
    for task in reversed(list(tasks.keys())):
        for dep in adj_list[task]:
            latest_finish[dep] = min(latest_finish[dep], latest_start[task])
            latest_start[dep] = latest_finish[dep] - tasks[dep][1]
    
    return latest_start, latest_finish

def find_critical_path(earliest_start, latest_start):
    critical_path = []
    for task in earliest_start:
        if earliest_start[task] == latest_start[task]:
            critical_path.append(task)
    return critical_path

def generate_daily_schedule(tasks, earliest_start, earliest_finish, start_date):
    worker_schedule = defaultdict(list)
    max_daily_hours = 8
    
    for task, (worker, duration) in tasks.items():
        start = earliest_start[task]
        remaining_hours = duration
        
        current_hour = start
        while remaining_hours > 0:
            hours_worked = min(remaining_hours, max_daily_hours - (current_hour % max_daily_hours))
            day = int((current_hour // max_daily_hours) + 1)  # Start days from 1 instead of 0, convert to int
            task_date = start_date + timedelta(days=day-1)  # Convert day to date
            worker_schedule[worker].append((task_date, task, int(hours_worked)))  # Ensure hours worked is an integer
            remaining_hours -= hours_worked
            current_hour += hours_worked
    
    # Sort the schedule by date in reverse order (earliest dates first)
    for worker in worker_schedule:
        worker_schedule[worker].sort(key=lambda x: x[0], reverse=False)
    
    return worker_schedule

def critical_path_method(tasks, dependencies, start_date, deadline):
    earliest_start, earliest_finish = calculate_earliest_times(tasks, dependencies)
    project_duration = int(max(earliest_finish.values()))  # Ensure project duration is an integer
    latest_start, latest_finish = calculate_latest_times(tasks, dependencies, project_duration)
    critical_path = find_critical_path(earliest_start, latest_start)
    worker_schedule = generate_daily_schedule(tasks, earliest_start, earliest_finish, start_date)
    
    project_end_date = start_date + timedelta(days=(project_duration // 8) - 1)  # Calculate end date based on duration

    print("Project Deadline:", deadline.strftime('%d/%m/%y'))
    print("Project Duration:", project_duration, "hours")
    print("Project Start Date:", start_date.strftime('%d/%m/%y'))
    print("Project End Date:", project_end_date.strftime('%d/%m/%y'))

    if project_end_date <= deadline:
        print("It is feasible and achievable")
    else:
        print("Need more resources and time")
    
    print("\nWorker Schedules:")
    for worker, schedule in worker_schedule.items():
        print(f"Schedule for {worker}:")
        for date, task, hours in schedule:
            print(f"  {date.strftime('%d/%m/%y')}: {task} ({hours} hours)")

common_dependencies = {
    "Analisis Kebutuhan": [],
    "Desain UI/UX": ["Analisis Kebutuhan"],
    "Perancangan Basis Data": ["Analisis Kebutuhan"],
    "Pembuatan Basis Data": ["Perancangan Basis Data"],
    "Frontend Development": ["Desain UI/UX"],
    "Backend Development": ["Perancangan Basis Data"],
    "Pengembangan API": ["Backend Development"],
    "Integrasi API": ["Pengembangan API"],
    "Pengujian Unit": ["Frontend Development", "Backend Development"],
    "Pengujian Integrasi": ["Integrasi API", "Frontend Development", "Backend Development"],
    "Integrasi Model": ["Integrasi API"],
    "Pengujian Sistem": ["Pengujian Integrasi"],
    "Pengujian Fungsionalitas": ["Pengujian Sistem"],
    "Pengujian User Acceptance (UAT)": ["Pengujian Fungsionalitas"],
    "Pengujian dan Perbaikan": ["Pengujian User Acceptance (UAT)"],
    "Evaluasi Model": ["Integrasi Model"],
    "Pembersihan dan Preprocessing Data": ["Pengumpulan Data"],
    "Pengumpulan Data": [],
    "Visualisasi Data": ["Pembersihan dan Preprocessing Data"],
    "Implementasi Fitur": ["Frontend Development", "Backend Development"],
    "Dokumentasi": ["Implementasi Fitur", "Frontend Development", "Backend Development"],
    "Deployment": ["Pengujian dan Perbaikan", "Frontend Development", "Backend Development", "Desain UI/UX"],
    "Presentasi dan Demo": ["Deployment"]
}

def apply_common_dependencies(predicted_labels):
    predicted_labels = [str(label) for label in predicted_labels]
    unique_labels = [f"{label}_{idx}" for idx, label in enumerate(predicted_labels)]
    label_to_unique = dict(zip(predicted_labels, unique_labels))
    dependencies = {}
    for task, deps in common_dependencies.items():
        if task in label_to_unique:
            unique_task = label_to_unique[task]
            unique_deps = [label_to_unique[dep] for dep in deps if dep in label_to_unique]
            dependencies[unique_task] = unique_deps
    return dependencies


# In[20]:


# Example usage
new_tasks = [
    "Mengimplementasikan tampilan percakapan yang menarik dan informatif, serta elemen-elemen interaktif seperti tombol dan formulir.", 
    "Membuat sistem moderasi konten untuk platform media sosial.",
    "Mengamankan akses ke platform cloud dan aplikasi web dengan menerapkan autentikasi dan otorisasi yang sesuai.",
    "Membuat style guide yang lengkap dengan komponen UI yang dapat digunakan ulang untuk aplikasi mobile perusahaan."
]

task_workers = ["UserID1", "UserID2", "UserID1", "UserID1"]

predicted_labels = predict_task_labels(loaded_model, tokenizer, label_encoder, new_tasks)
tasks = assign_workers_to_tasks(predicted_labels, task_workers)
dependencies = apply_common_dependencies(predicted_labels)

# Define the start date for the schedule
start_date = datetime(2024, 6, 10)
deadline = datetime(2024, 12, 12)

# Run the CPM algorithm with user inputs
critical_path_method(tasks, dependencies, start_date, deadline)

