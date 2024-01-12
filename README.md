## Disaster Response Pipeline Project
The objective of the project is to build a classification model that categorizes text messages into 36 categories. The short text messages, which are sent by people during a natural disaster require different follow up action depending on the information in the message. When a disaster happenes a lot of messages are sent. Therefore a good model, which can classify messages into appropriate categories, will improve the process of help provided to people.
It is a supervised learning classification task. The project code is written in Python language. 
The following libraries are needed to be installed as a prerequisite:
* json
* plotly
* pandas
* nltk
* flask
* sklearn
* sqlalchemy
* sys
* pickle

### Files in the repository
```
app
|- templates
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # code to process the data
|- DisasterResponse.db # database to save clean data to
models
|- train_classifier.py # code to build, evaluate and save a classification model
|- classifier.pkl # saved model
README.md
```

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
