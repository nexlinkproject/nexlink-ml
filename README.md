# NEXLINK - MACHINE LEARNING TEAM REPOSITORY

## Description

This repository contains the code and resources for training machine learning text classification model, scheduling algorithm, and feedback learning for Nexlink application. This repository also contain Feedback_learning API

### Directory

- notebook_training : Contains scratch notebook files that we have developed to train the model and merge to scheduling algorithm and feedback learning
- Notebook : Contains python files for feedback learning API
- dataset : contains dataset csv file for the models
- model : contains exported files (.h5, .pkl)
- monitor : contains cloud function deployment to trigger the retraining script model

### Machine Learning Overall Model Diagram

![Model Diagram](/model-diagram.png)

It start with understanding the data, preprocessing data, train the model, evaluatte and improve, load the model in the scheduling algorithm, and feedback learning to improve the model.

## Step to train the model
1. Open the notebook_training file which consists of all of the model and algorithms code. Also preapre the dataset which in dataset folder
2. Fork or download it so you can run it locally
3. Prepare your IDE (colab, jupyter, etc.)
4. Make sure the directory of the dataset is correct
5. Run the text-classification-model which automatically save your model, .h5, and pkls
6. you can then proceed the scheduling algorithm and try with different output
7. About the API of this scheduling and text classification model is on ml-api repository

## Step to build the dockerfile

dockerfile for feedbacklearning algorithm

### 1. Build the DockerFile

```
docker build -t feedback-learning-api .
```

### 2. Run the docker

```
docker run -d --name feedback-learning-container -p 8080:8080 feedback-learning-api
```

### 3. Check the logs

```
docker logs feedback-learning-container
```

