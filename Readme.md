# Disaster Response Pipeline Project

### Summary:

I've gained and developed my data engineering abilities in this course, which will increase my possibilities and potential as a data scientist. In order to create a model for an API that categorizes catastrophe messages, I'll use these abilities in this project to evaluate disaster data from Appen.

# Acknowledgements

### Installations

`pip install -r requirements.txt`

### File in the repository

```
app
| - template
| |- master.html
| |- go.html
|- run.py
data
|- disaster_categories.csv
|- disaster_messages.csv
|- process_data.py
|- InsertDatabaseName.db
models
|- train_classifier.py
|- classifier.pkl
README.md
```

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/
