# Disaster Response Pipeline Project

This repository contains a disaster response pipeline, which is used to classify and categorize messages related to natural disasters. The main objective of the pipeline is to facilitate the rapid identification and routing of relevant messages to the appropriate organizations for a more effective response to emergency situations.

## Repository contents

The repository includes the following files and folders:

*data: folder containing the data used to train and test the model.
*models: Folder containing the trained models. 
*app: Folder containing a web application to interact with the model and classify new messages.
*README.md: This documentation file.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Contributions
Contributions to this project are welcome. If you encounter any problems, have any ideas for improvement or would like to add new features, please feel free to do so. 

