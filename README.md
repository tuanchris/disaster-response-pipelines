# Disaster Response Pipeline Project
1. Project motivation
2. Installation
3. File descriptions
4. Instructions

## Project motivation
This project attempts to build data pipelines that take in free-text messages that are scraped from social network during disaster times, and classify them using 36 pre-defined categories. Pipelines like this can help governments and organizations quickly response to disasters and focus their attention where they are needed the most.

This project will build an ETL pipeline that read in messages and their label categories, clean and transform them for ML use and load them to a SQLite database. Then a machine learning pipeline will take over to load in data, engineer features and train machine learning model. Trained model that has been optimized will be save to a file. A web application will allow user to use the trained model by predicting categories for user input messages.

The personal motivation for this project is to finish a full work flow of a data scientist. The following steps are all part of this project:
* Getting data from multiple sources
* ETL data
* EDA
* Feature engineering and extraction
* Model selection and evaluation
* Model tuning
* Building an user facing app

## Installation
Create virtual environment and install required packages
```
conda create -n disaster-response
conda activate disaster-response
pip install -r requirements.txt
```


## File descriptions
1. app
  * run.py: Flask based web app for user interaction and model serving
  * template: template to render the app
2. data
  * disaster_categories.csv: label categories of messages
  * disaster_messages.csv: raw messages with genre
  * DisasterResponse.db: output of ETL pipeline (sqlite db)
  * process_data.py: ETL pipelines that get and clean data
3. models
  * classifier.pkl: pickle model, output of machine learning model
  * train_classifier.py: machine learning pipeline script
4. notebooks
  * notebooks to help build ETL and ML pipelines

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
