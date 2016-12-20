#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder

#Read training data
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

#features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]
training_label = dataset['Survived']

#Feature Engineering
training_data = training_data.fillna(0)
#training_data['Name'] = LabelEncoder().fit_transform(training_data['Name'])
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])
#training_data['Ticket'] = LabelEncoder().fit_transform(training_data['Ticket'])
training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'])
training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'])

model = RandomForestClassifier()
model.fit(training_data, training_label)

y_pos = np.arange(len(features))
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, features)
plt.xlabel('features')
plt.title('feature_importances')
#plt.show()

#Train model
features = ['Sex', 'Age', 'Fare']
training_data = dataset[features]
training_data = training_data.fillna(0)
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])
model = RandomForestClassifier()
model.fit(training_data, training_label)

test_data = pd.read_csv(SCRIPT_PATH + "/test.csv")
preidction_data = test_data[features]
preidction_data = preidction_data.fillna(0)
preidction_data['Sex'] = LabelEncoder().fit_transform(preidction_data['Sex'])
result_lables = model.predict(preidction_data)
results = pd.DataFrame({
    'PassengerId' : test_data['PassengerId'],
    'Survived' : result_lables
})

results.to_csv(SCRIPT_PATH + "/submission2.csv", index=False)


cv_scores = np.mean(cross_val_score(model, training_data, training_label, scoring='roc_auc', cv=5))
print cv_scores
