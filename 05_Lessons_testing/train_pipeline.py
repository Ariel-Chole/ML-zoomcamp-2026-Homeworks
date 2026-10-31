#importing necessary libararies

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

#importing dataset
df = pd.read_csv("course_lead_scoring_2026.csv")

#building pipeline
categorical = ['lead_source']
numeric = ['number_of_courses_viewed', 'annual_income']

df[categorical] = df[categorical].fillna('NA')
df[numeric] = df[numeric].fillna(0)

train_dict = df[categorical + numeric].to_dict(orient='records')
y_train = df.converted.values
pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    LogisticRegression(solver='liblinear')
)

pipeline.fit(train_dict, y_train)


# Save the model
import pickle

output_file = f"model_pipeline.bin"

with open(output_file,"wb") as f_out:
    pickle.dump(pipeline,f_out)






