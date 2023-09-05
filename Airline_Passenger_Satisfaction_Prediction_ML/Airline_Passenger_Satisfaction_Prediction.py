# Libraries & Frameworks
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from sklearn.metrics import accuracy_score


# Data Extraction From Zip
with ZipFile('satisfaction.zip') as data_zip:
    data_zip.extractall('data')


# Constants
data_train = 'data/train.csv'
data_test = 'data/test.csv'
SPLIT = 0.25


# Data Preprocessing
df_train = pd.read_csv(data_train)
df_test = pd.read_csv(data_test)

df = pd.concat([df_train, df_test])

print(df.info())

print(df.isnull().sum())

df = df.dropna()

print(df.isnull().sum())

print(df.info())

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    elif df[col].dtype != 'object':
        pass

print(df.dtypes)


# Train Test Split
features = df.drop('satisfaction', axis= 1)
target = df['satisfaction']

X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                   test_size= SPLIT,
                                                   shuffle= True,
                                                   random_state= 24
                                                   )


print(X_train.shape,
      Y_train.shape,
      X_test.shape,
      Y_test.shape,
      '\n'
      )


# Model Training
m = DecisionTreeClassifier()

m.fit(X_train, Y_train)


# Model Testing Function
def Model_accuracy(model, X_train, Y_train, X_test, Y_test):
    print("MODEL TESTING : \n")
    print(f'{model}')

    pred_train = model.predict(X_train)
    print(f'Training Accuracy is : {accuracy_score(Y_train, pred_train)}')

    pred_test = model.predict(X_test)
    print(f'Testing Accuracy is : {accuracy_score(Y_test, pred_test)}\n')


# Testing
Model_accuracy(model= m,
               X_train= X_train,
               Y_train= Y_train,
               X_test= X_test,
               Y_test= Y_test
               )