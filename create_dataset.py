# Create a toy dataset for classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# create a toy dataset
df = pd.DataFrame({
    "x1": np.random.randint(0, 100, 10000),
    "x2": np.random.randint(0, 100, 10000),
    "x3": np.random.randint(0, 100, 10000),
    "x4": np.random.randint(0, 100, 10000),
    "survived": np.random.randint(0, 2, 10000)
})


# save to csv
df.to_csv("data/data.csv", index=False)
