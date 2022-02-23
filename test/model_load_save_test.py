import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.NeuralGAM.ngam import NeuralGAM, load_model


class ModelLoadSaveTests(unittest.TestCase):
    """Test for model loading/saving with dill"""

    def tests_save_load_model(self):
        
        # Generate training data (random values between -50 and 50 )
        x1 = np.array(-10 + np.random.random((10))*10)
        x2 = np.array(-10 + np.random.random((10))*10)
        x3 = np.array(-10 + np.random.random((10))*10)
        b = np.array(-10 + np.random.random((10))*10)

        X = pd.DataFrame([x1,x2,x3, b]).transpose()

        # y = f(x1) + f(x2) + f(x3) =  x1^2 + 2x2 + sin(x3).
        y = pd.Series(x1 + x2 + x3 + b)

        # use data split and fit to run the model
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        ngam = NeuralGAM(num_inputs = len(x_train.columns), num_units=64)

        ycal, mse = ngam.fit(x_train, y_train, 1, None, 1)
        y_pred = ngam.predict(x_test)
        
        # serialize model to file
        file = "./output.model"
        ngam.save_model(file)
        
        #reload file
        new_gam = load_model(file)
        self.assertIsNotNone(new_gam)
        
        new_pred = new_gam.predict(x_test)
        self.assertEqual(y_pred.all(), new_pred.all())
        
