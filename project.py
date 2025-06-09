
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('mail_data.csv')
print(data.head(), '\n')
print(data.shape, '\n')
print(data.describe(), '\n')


# Replace the null values with a null string -

mail_data = data.where((pd.notnull(data)),'')
print(mail_data.head(),'\n')


# Label Encoding -
# ( Label spam mail as 0, and ham mail as 1)

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# Separating the data as texts and label -

X = mail_data['Message']
Y = mail_data['Category']
print(X, '\n')
print(Y, '\n')


# Splitting the data into Training data and Test Data -

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=3)
print(X.shape, X_train.shape, X_test.shape)


# Feature Extraction -
# (Transform the text data to feature vectors that can be used as input to the logistic regression)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)


# Convert Y_train and Y_test values as Integers -

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train, '\n')
print(X_train_feature, '\n')
print(X_test_feature, '\n')
print(Y_train, '\n')
print(Y_test, '\n')


# Training the model -
# Logistic Regression

model = LogisticRegression()
model.fit(X_train_feature, Y_train)

# Prediction on training data -

prediction_on_training_data = model.predict(X_train_feature)
Accurancy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accurancy on training data :',Accurancy_on_training_data, '\n')

# Prediction on training data -

prediction_on_testing_data = model.predict(X_test_feature)
Accurancy_on_testing_data = accuracy_score(Y_test, prediction_on_testing_data)
print('Accurancy on testing data :',Accurancy_on_testing_data, '\n')


# Building a predictive data -

input_mail = ["Nah I don't think he goes to usf, he lives around here though"]

# Convert text to feature vectors -
input_data_feature = feature_extraction.transform(input_mail)

#Making Prediction -
prediction = model.predict(input_data_feature)
print(prediction, '\n')

if prediction[0] == 1 :
    print('Your mail has HAM Mail')
else:
    print('Your mail has SPAM Mail')