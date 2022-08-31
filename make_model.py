"""
This file generates and selects a suitable classifier for our question.
We will use the trained classifier to implement out API
"""
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# load data
df = pd.read_csv("data/data.csv")
y = df["survived"]
X = df.drop("survived", axis=1)


# train/test split, shuffle in real life
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

assert X_train.shape[0] == y_train.shape[0]

# make model
clf1 = GradientBoostingClassifier()
clf2 = RandomForestClassifier()
meta_clf = LogisticRegression()
models = [clf1, clf2]

# train models
for clf in models:
    clf.fit(X_train, y_train)

# make stacking model
meta_X_train = np.column_stack(model.predict(X_train) for model in models)
meta_X_test = np.column_stack(model.predict(X_test) for model in models)

# train meta model
meta_clf.fit(meta_X_train, y_train)
score = meta_clf.score(meta_X_test, y_test)
print("Score:", score)
print("REPORT:\n", classification_report(
    y_test, meta_clf.predict(meta_X_test)))

# Retraining model on entire data set

for clf in models:
    clf.fit(X, y)

meta_X = np.column_stack(model.predict(X) for model in models)
meta_clf.fit(meta_X, y)

# Saving the model
dump(meta_clf, "./model/stacked_model.bin")
