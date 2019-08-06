# Disaster Response Pipeline Project
1. Installation
2. Project motivation
3. File descriptions
4. Instructions

## Installation
Create virtual environment and install required packages
```
conda create -n disaster-response
conda activate disaster-response
pip install -r requirements.txt
```
## Project motivation
The motivation for this project is to finish a full work flow of a data scientist. The following steps are all part of this project:
* Getting data from multiple sources
* ETL data
* EDA
* Feature engineering and extraction
* Model selection and evaluation
* Model tuning
* Building an user facing app

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
