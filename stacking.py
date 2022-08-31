import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel, conlist


# Define object we classify
class ObjectInformation(BaseModel):
    x1: int
    x2: int
    x3: int
    x4: int

# Create the Stack model


class stackingModel():
    """
    Implementation of the Stacking model from the paper.
    """

    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit_models(self, X, y):
        for clf in self.models:
            clf.fit(X, y)

        meta_X = np.column_stack(
            model.predict(X) for model in self.models)

        fitted_meta_model = self.meta_model.fit(meta_X, y)
        return fitted_meta_model

    def predict_value(self, item: ObjectInformation, fitted_meta_model):
        item = np.array([item.x1, item.x2, item.x3, item.x4])
        item = item.reshape(1, -1)

        meta_X = np.column_stack(
            model.predict(item) for model in self.models)

        prediction = fitted_meta_model.predict(meta_X)
        probability = fitted_meta_model.predict_proba(meta_X).max()

        return prediction, probability


#! ###############################
#! TRAINING OF THE MODEL
#! ###############################
# Load data
data = pd.read_csv("data/data.csv")
y = data["survived"]
X = data.drop("survived", axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# make models
clf1 = GradientBoostingClassifier()
clf2 = RandomForestClassifier()
meta_clf = LogisticRegression()
models = [clf1, clf2]

# train models
for model in models:
    model.fit(X_train, y_train)

# make stacking model
meta_X_train = np.column_stack(
    model.predict(X_train) for model in models)

meta_X_test = np.column_stack(
    model.predict(X_test) for model in models)

# fit meta model
fitted_meta_model_train = meta_clf.fit(meta_X_train, y_train)
score = fitted_meta_model_train.score(meta_X_test, y_test)

print("Score:", score)
print("REPORT:\n", classification_report(
    y_test, meta_clf.predict(meta_X_test)))

# Retraining model on entire data set

for model in models:
    model.fit(X, y)

meta_X = np.column_stack(
    model.predict(X) for model in models)

fitted_meta_model = meta_clf.fit(meta_X, y)


#! ###############################
#! DEVELOPMENT OF API
#! ###############################

# Create the FastAPI app
app = FastAPI()
predictor = stackingModel(models, meta_clf)


@app.get("/")
def root():
    return {"GoTo": "/docs"}

# Create the predict endpoint


@app.post("/predict")
async def predict(item: ObjectInformation):
    try:
        prediction, probability = predictor.predict_value(
            item, fitted_meta_model)
        return {"prediction": int(prediction[0]), "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
