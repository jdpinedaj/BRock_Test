import unittest
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from stacking import StackingModel, ObjectInformation


class TestStackingModel(unittest.TestCase):
    def test_upload_file(self):
        """
        Test if the file is uploaded.
        """
        df = pd.read_csv("./data/data.csv")
        self.assertIsInstance(df, pd.DataFrame)

    def test_fit_and_predict_value(self):
        """
        Test if the model is fitted and the prediction is correct.
        """

        df = pd.read_csv("./data/data.csv")

        # make models
        clf1 = GradientBoostingClassifier()
        clf2 = RandomForestClassifier()
        meta_clf = LogisticRegression()

        # make stacking model
        stack = StackingModel
        obj = ObjectInformation(x1=1, x2=2, x3=3, x4=4)

        probability = stack.fit_and_predict(clf1, clf2, meta_clf, df, obj)

        self.assertIsInstance(probability, float)


if __name__ == "__main__":
    unittest.main()
