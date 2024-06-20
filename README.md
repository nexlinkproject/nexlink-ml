# NEXLINK - MACHINE LEARNING REPO

## Description
This repository contains the code and resources for training machine learning text classification model, scheduling algorithm, and feedback learning for Nexlink application. 

### Directory
- Notebook : Contains python files for the model, scheduling algorithm, and feedback learning
- dataset : contains dataset csv file for the models
- model : contains exported files (.h5, .pkl)

## 1. Navigate and Get The Dataset File

```
cd dataset
```

## 2. Installation
You will need certain libraries, navigate to Notebook directory and install

```
cd /
cd Notebook
pip install -r requirements.txt
```

## 3. Run The Main File

```
python firstModel.py
```

## 4. Run The Scheduling Algorithm

```
python schedule_algorithms.py
```

## 5. Run The Feedback Learning Algorithm
This model is to improve the model overtime while connected to postgre SQL

```
python feedback_learning_Final.py
```


