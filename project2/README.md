Course_data_science
#Project2: Disaster Response

There are three parts of code in this projects.
- data: data_process.py to process the raw data before training the model
- model: train_classifier.py to build the model and saving the model
- app: run.py to run the web app and deliver the parameters to html files

#how to use
- first of all, run data_processing.py to load the rawdata.Then the rawdata will be cleaned and saved into database.
- secondly, run train_classifier.py to load the clean data from database. And the model will be trained through the pipeline which is defined in the code.
- last of all, run run.py to process the data and predict the data by the model trained before to plot the result in html files.

#the lib imported in this project
- data_process.py: pandas, sqlalchemy, sys
- train_classifier.py: sys, sqlalchemy, pandas, re, nltk, sklearn
- run.py: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy