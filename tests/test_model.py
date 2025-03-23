import unittest
import numpy as np
import pandas as pd
from neuralGAM.model import NeuralGAM

class TestNeuralGAM(unittest.TestCase):

    def setUp(self):
        # Create sample data
        np.random.seed(0)
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.y_train = 3 * self.X_train['feature1'] + 2 * self.X_train['feature2'] + np.random.randn(100)
        self.w_train = np.random.rand(100)

    def test_fit_gaussian(self):
        model = NeuralGAM(family='gaussian')
        y, g, eta = model.fit(self.X_train, self.y_train)
        self.assertEqual(len(y), len(self.y_train))
        self.assertEqual(len(g), len(self.X_train))
        self.assertEqual(len(eta), len(self.y_train))

    def test_fit_binomial(self):
        y_train_binomial = (self.y_train > np.median(self.y_train)).astype(int)
        model = NeuralGAM(family='binomial')
        y, g, eta = model.fit(self.X_train, y_train_binomial)
        self.assertEqual(len(y), len(y_train_binomial))
        self.assertEqual(len(g), len(self.X_train))
        self.assertEqual(len(eta), len(y_train_binomial))

    def test_fit_parallel(self):
        model = NeuralGAM(family='gaussian', verbose=0)
        y, g, eta = model.fit(self.X_train, self.y_train, parallel=True)
        self.assertEqual(len(y), len(self.y_train))
        self.assertEqual(len(g), len(self.X_train))
        self.assertEqual(len(eta), len(self.y_train))

    def test_fit_sequential(self):
        model = NeuralGAM(family='gaussian', verbose=0)
        y, g, eta = model.fit(self.X_train, self.y_train, parallel=False)
        self.assertEqual(len(y), len(self.y_train))
        self.assertEqual(len(g), len(self.X_train))
        self.assertEqual(len(eta), len(self.y_train))

    def test_fit_with_linear_terms(self):
        model = NeuralGAM(family='gaussian', linear_terms=['feature1'])
        y, g, eta = model.fit(self.X_train, self.y_train)
        self.assertEqual(len(y), len(self.y_train))
        self.assertEqual(len(g), len(self.X_train))
        self.assertEqual(len(eta), len(self.y_train))

if __name__ == '__main__':
    unittest.main()