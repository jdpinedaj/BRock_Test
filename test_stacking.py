import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from stacking import stackingModel, ObjectInformation


class TestStackingModel(unittest.TestCase):

    def test_fit_models(self):

        df = pd.read_csv("data/data.csv")
        y = df["survived"]
        X = df.drop("survived", axis=1)

        # make models
        clf1 = GradientBoostingClassifier()
        clf2 = RandomForestClassifier()

        # make meta model
        meta_clf = LogisticRegression()

        # make stacking model
        stack = stackingModel([clf1, clf2], meta_clf)

        # fit
        stack.fit_models(X, y)

        # assert
        self.assertIsInstance(stack, stackingModel)

    def test_predict_value(self):

        # data
        data = pd.read_csv("data/data.csv")
        y = data["survived"]
        X = data.drop("survived", axis=1)

        # make models

        models = [GradientBoostingClassifier(), RandomForestClassifier()]
        meta_model = LogisticRegression()
        self.predictor = stackingModel(models, meta_model)

        fitted_model = self.predictor.fit_models(X, y)

        # Object to predict
        obj = ObjectInformation(x1=11, x2=20, x3=3, x4=14)

        # predict
        prediction, probability = self.predictor.predict_value(
            obj, fitted_model)

        # assert
        self.assertIsInstance(prediction, int)
        self.assertIsInstance(probability, np.ndarray)


if __name__ == "__main__":
    unittest.main()
