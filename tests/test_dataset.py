import unittest
import numpy as np
import pandas as pd

from neuralGAM.dataset import split, generate_err, get_truncated_normal, generate_normal_data, generate_uniform_data, compute_y, generate_data

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        self.y = pd.Series(np.random.randn(100))
        self.fs = pd.DataFrame({
            'feature1_transformed': self.X['feature1'] ** 2,
            'feature2_transformed': 2 * self.X['feature2'],
            'feature3_transformed': np.sin(self.X['feature3'])
        })
        self.output_folder = 'test_output'

    def test_split(self):
        X_train, X_test, y_train, y_test, fs_train, fs_test = split(self.X, self.y, self.fs)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        self.assertEqual(len(fs_train), 80)
        self.assertEqual(len(fs_test), 20)


    def test_generate_err(self):
        err = generate_err(100, "homoscedastic", self.y)
        self.assertEqual(len(err), 100)
        self.assertTrue(np.allclose(np.mean(err), 0, atol=0.1))

    def test_get_truncated_normal(self):
        data = get_truncated_normal(mean=0, sd=1, low=-2, upp=2, nrows=100)
        self.assertEqual(len(data), 100)
        self.assertTrue(np.all(data >= -2))
        self.assertTrue(np.all(data <= 2))

    def test_generate_normal_data(self):
        X, y, fs = generate_normal_data(100, "homoscedastic", "gaussian")
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(fs), 100)

    def test_generate_uniform_data(self):
        X, y, fs = generate_uniform_data(100, "homoscedastic", "gaussian")
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(fs), 100)

    def test_compute_y(self):
        y = compute_y(self.fs, np.ones(100) * 2, 100, "homoscedastic", "gaussian")
        self.assertEqual(len(y), 100)

    def test_generate_data(self):
        X, y, fs = generate_data("homoscedastic", "normal", "gaussian", 100)
        self.assertEqual(len(X), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(len(fs), 100)

if __name__ == '__main__':
    unittest.main()