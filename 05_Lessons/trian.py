#importing necessary libraries
import pickle

# import tqdm
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold

#parameters 
C = 1.0
n_folds = 5

#loading data
df = pd.read_csv("course_lead_scoring_2026.csv")

#Data preparation
def data_prepa(df):
    categorical_features = list((df.dtypes[df.dtypes == 'str']).index)
    numerical_features = list((df.dtypes[df.dtypes != 'str']).index)

    return categorical_features,numerical_features

#preparing dataset
categorical_features,numerical_features = data_prepa(df)

"""
If there are missing values:
For categorical features, replace them with 'NA'
For numerical features, replace with with 0.0 
"""
df[categorical_features] = df[categorical_features].fillna("NA")

df[numerical_features] = df[numerical_features].fillna(0)

#Splitting the data into 3 parts, train, test, val
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

#building train function 
def train_model(df_train,C):
    y_train = df_train.converted.values
    X_train = df_train[categorical_features+numerical_features].reset_index(drop=True)\
        .to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    train_dicts = dv.fit_transform(X_train)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(train_dicts,y_train)

    return model,dv


#bulding predict function
def predict(df_val,model,dv):
    X = df_val[categorical_features+numerical_features].reset_index(drop=True)\
    .to_dict(orient="records")

    X_val = dv.transform(X)
    y_pred = model.predict_proba(X_val)[:,1]
    y_val = df_val.converted.values

    return y_pred,y_val

#validating the dataset by KFold
print(f"Doing validation with C = {C}")
fold = 1
kfold = KFold(n_splits=n_folds,shuffle=True,random_state=1)

scores = []
for train_idx,val_idx in kfold.split(df_full_train):
    #separating train and val dataset by each fold
    fold_train = df_full_train.iloc[train_idx]
    fold_val = df_full_train.iloc[val_idx]

    model,dv = train_model(fold_train,C = C)
    fold_predictions,y_val = predict(fold_val,model,dv)

    auc = roc_auc_score(y_val,fold_predictions)
    scores.append(auc)

    print("aucon fold {} is {}.".format(fold,auc))

    fold += 1
print("Validation Result: ")
print('C=%s %.3f +- %.3f' % (C,np.mean(scores),np.std(scores)))

# Save the model

# output_file = "model_C = %s.bin"%C
output_file = f"model_C = {C}.bin"
output_file


# f_out = open(output_file,"wb")
# pickle.dump((model,dv),f_out)
# f_out.close()#don't forget to close file

with open(output_file,"wb") as f_out:
    pickle.dump((model,dv),f_out)

print(f"Model is saved successfully!!")