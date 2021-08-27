# Disaster Response Pipeline Project
## Table of Contents
1. [Project Overview](#-Project-Overview)
2. [Project Components](#-Project-Components)
3. [Installation](#-Installation)
4. [liscense](#-liscense)
## 1. [Project Overview](#-Project-Overview)
This project is part of the [Udacity Data Science Nano Degree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)and supported by [Appen](https://appen.com/). This project will analyze a [data set](https://github.com/petitblue/Udacity_Data_Science/tree/main/Project%202%20Disaster%20Response%20Pipeline/data) containing real messages that were sent during disaster events. Those messages are sent from social media or from disaster response organizations. This project will build a ETL pipeline to load and process data, and a machine learning pipeline to classify those messages so as to send them to an appropriate disaster relief agency.
## 2. [Project-Components]
There are three components in the project.
### 1. ETL Pipeline
Loads the message.csv and categories.csv files and merges two datasets
clean data and stores it in a SQLite database
### 2. ML Pipeline
Build a test processing and maching learning pipline
Train and tunes a model using GridSearchCV
### 3. Flask Web App
There is a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


## 2. [Installation]()
### Devendencies :
   - [python (>=3.6)](https://www.python.org/downloads/)  
   - [pandas](https://pandas.pydata.org/)  
   - [numpy](https://numpy.org/)  
   - [sqlalchemy](https://www.sqlalchemy.org/)  
   - [nltk](https://www.nltk.org/)  
   - [sys](https://docs.python.org/3/library/sys.html)  
   - [plotly](https://plotly.com/python/)  
   - [sklearn](https://sklearn.org/)  
   - [joblib](https://joblib.readthedocs.io/en/latest/)  
   - [flask](https://flask.palletsprojects.com/en/2.0.x/)  
   - [WordCloud](https://pypi.org/project/wordcloud/)
 ### Download and Installation
 git clone https://github.com/petitblue/udacity_data_science/project2-disaster-response-pipeline
 
 While in the project's root directory disaster-response-pipeline run the ETL pipeline that cleans and stores data in database.
 
 python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

 Next, run the ML pipeline that trains the classifier and save it.
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Next, change directory into the app directory and run the Python file run.py.
cd app
python run.py
Finally, go to http://0.0.0.0:3001/ or http://localhost:3001/ in your web-browser.
Type a message input box and click on the Classify Message button to see the various categories that your message falls into.
## 3. [liscense]()

