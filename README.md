# Disaster Response Pipeline Project

This repository contains a disaster response pipeline, which is used to classify and categorize messages related to natural disasters. The main objective of the pipeline is to facilitate the rapid identification and routing of relevant messages to the appropriate organizations for a more effective response to emergency situations.

## Repository contents

The repository includes the following files and folders:

* data: folder containing the data used to train and test the model.
* models: Folder containing the trained models. 
* app: Folder containing a web application to interact with the model and classify new messages.
* README.md: This documentation file.

## Requirements
To run the pipeline and the web application, the following dependencies are required:
Python 3.7 or higher
Python libraries: pandas, numpy, scikit-learn, nltk, flask.

## Usage
The pipeline consists of the following steps:

Data preprocessing: Text data is cleaned and processed, including tokenization, stop word removal, lemmatization, and normalization.
Feature extraction: Relevant features are extracted from the preprocessed text data using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
Model training: A machine learning model is trained using the labeled data.
Message classification: The trained model is used to classify new messages and assign them to appropriate categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Web Application
The web application in the app folder allows you to interact with the trained model and classify new messages through a graphical interface. To run the application, simply execute the app/run.py file 

## Contributions
Contributions to this project are welcome. If you find any issues, have any improvement ideas, or would like to add new features, please feel free to do so. 

## Credits
This pipeline was developed as part of the final project for the Udacity Data Science for Disaster Response course. The data used in the pipeline was provided by Udacity.
