import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from stacking import stackingModel, ObjectInformation


class TestStackingModel(unittest.TestCase):
    def test_upload_file(self):
        df = pd.read_csv("./data/data.csv")
        self.assertIsInstance(df, pd.DataFrame)

    def test_fit_and_predict_value(self):

        df = pd.read_csv("./data/data.csv")
        y = df["survived"]
        X = df.drop("survived", axis=1)

        # make models
        clf1 = GradientBoostingClassifier()
        clf2 = RandomForestClassifier()
        meta_clf = LogisticRegression()

        # make stacking model
        stack = stackingModel
        obj = ObjectInformation(x1=1, x2=2, x3=3, x4=4)

        probability = stack.fit_and_predict(clf1, clf2, meta_clf, df, obj)

        self.assertIsInstance(probability, float)


if __name__ == "__main__":
    unittest.main()
