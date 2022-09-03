import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from fastapi import FastAPI, File, UploadFile, Depends
from pydantic import BaseModel
from enum import Enum


# Classes
class Classifier(str, Enum):
    lr = "lr"
    knn = "knn"
    dt = "dt"
    rf = "rf"
    gb = "gb"


class ObjectInformation(BaseModel):
    x1: int
    x2: int
    x3: int
    x4: int


data = pd.read_csv("./data/data.csv")


class StackingModel:
    """
    Implementation of the Stacking model from the paper.
    """

    def upload_file(file: UploadFile = File(...)):
        """
        Upload file to the server.
        """
        df = pd.read_csv(file.file)
        file.file.close()

        return df

    def get_model_1(model1: Classifier):
        """
        Get the first model.
        """
        if model1 == Classifier.lr:
            first_model = LogisticRegression()
        elif model1 == Classifier.knn:
            first_model = KNeighborsClassifier()
        elif model1 == Classifier.dt:
            first_model = DecisionTreeClassifier()
        elif model1 == Classifier.rf:
            first_model = RandomForestClassifier()
        elif model1 == Classifier.gb:
            first_model = GradientBoostingClassifier()
        else:
            raise ValueError("Model not found")

        return first_model

    def get_model_2(model2: Classifier):
        """
        Get the second model.
        """
        if model2 == Classifier.lr:
            second_model = LogisticRegression()
        elif model2 == Classifier.knn:
            second_model = KNeighborsClassifier()
        elif model2 == Classifier.dt:
            second_model = DecisionTreeClassifier()
        elif model2 == Classifier.rf:
            second_model = RandomForestClassifier()
        elif model2 == Classifier.gb:
            second_model = GradientBoostingClassifier()
        else:
            raise ValueError("Model not found")

        return second_model

    def get_meta_model(meta_model: Classifier):
        """
        Get the meta model.
        """
        if meta_model == Classifier.lr:
            last_model = LogisticRegression()
        elif meta_model == Classifier.knn:
            last_model = KNeighborsClassifier()
        elif meta_model == Classifier.dt:
            last_model = DecisionTreeClassifier()
        elif meta_model == Classifier.rf:
            last_model = RandomForestClassifier()
        elif meta_model == Classifier.gb:
            last_model = GradientBoostingClassifier()
        else:
            raise ValueError("Model not found")

        return last_model

    def fit_and_predict(
        first_model, second_model, last_model, df, item: ObjectInformation
    ):
        """
        Fit and predict the model.
        """
        # Split the data
        y = df["survived"]
        X = df.drop("survived", axis=1)

        # fit models
        models = [first_model, second_model]
        for model in models:
            model.fit(X, y)
        meta_X = np.column_stack([model.predict(X) for model in models])

        # fit meta model
        fitted_meta_model = last_model.fit(meta_X, y)

        # Reading and transforming the data
        item = np.array([item.x1, item.x2, item.x3, item.x4])
        item = item.reshape(1, -1)
        meta_item = np.column_stack([model.predict(item) for model in models])

        # Predicting the data
        probability = fitted_meta_model.predict_proba(meta_item)[0][1]

        return probability


#! ###############################
#! TRAINING OF THE MODEL
#! ###############################
# Load data
data = pd.read_csv("data/data.csv")
y = data["survived"]
X = data.drop("survived", axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# make models
model1 = GradientBoostingClassifier()
model2 = RandomForestClassifier()
meta_model = LogisticRegression()

# train models
models = [model1, model2]
for model in models:
    model.fit(X_train, y_train)

# make stacking model
meta_X_train = np.column_stack([model.predict(X_train) for model in models])

meta_X_test = np.column_stack([model.predict(X_test) for model in models])

# fit meta model
fitted_meta_model_train = meta_model.fit(meta_X_train, y_train)
score = fitted_meta_model_train.score(meta_X_test, y_test)

print("Score:", score)
print("REPORT:\n", classification_report(y_test, meta_model.predict(meta_X_test)))

# Retraining model on entire data set
for model in models:
    model.fit(X, y)

meta_X = np.column_stack(model.predict(X) for model in models)
fitted_meta_model = meta_model.fit(meta_X, y)


#! ###############################
#! DEVELOPMENT OF API
#! ###############################

# Create the FastAPI app
app = FastAPI(
    title="Stacking API", description="API for the Stacking model", version="v1"
)
predictor = StackingModel


@app.get("/")
def root():
    """
    Root of the API.
    """
    return {"GoTo": "/docs"}


@app.post("/fit_and_predict")
async def fit_and_predict(
    model1: Classifier,
    model2: Classifier,
    meta_clf: Classifier,
    item: ObjectInformation = Depends(),
    file: UploadFile = File(...),
):
    """
    Fit the models and meta model and predict the value of the object.
    """
    # Uploading file
    df = predictor.upload_file(file)

    # Getting models
    first_model = predictor.get_model_1(model1)
    second_model = predictor.get_model_2(model2)
    last_model = predictor.get_meta_model(meta_clf)

    # Fitting and predicting
    probability = predictor.fit_and_predict(
        first_model, second_model, last_model, df, item
    )

    return {"Probability of surviving": round(probability, 2)}
