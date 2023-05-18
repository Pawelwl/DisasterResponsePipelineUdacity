# Disaster Response Pipeline Project
The project is a supervised learning classification task. It takes as input a text message and classifies it into the relvent number of possible 36 output categories.
The project code is written in Python language. THe following libraries are needed to be installed:
json
plotly
pandas
nltk
flask
sklearn
sqlalchemy
sys
pickle

### Instructions on how to run the app:
1. Download the code and install all necessary Python libraries.
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to see the web application.
