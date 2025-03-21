# Example usage of NeuralGAMMultinomial with the IRIS dataset

# Load the IRIS dataset
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from neuralGAM import NeuralGAMMultinomial


# Define parametric and nonparametric terms
p_terms = []  # No parametric terms in this example
np_terms = ['V1', 'V2', 'V3']  # Selected nonparametric terms
terms = p_terms + np_terms

# Load the Wine dataset
dataset = fetch_openml('volcanoes-b4', version=1)
X = dataset.data[terms]
y = dataset.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the NeuralGAMMultinomial model

n_classes = y.nunique()

model = NeuralGAMMultinomial(p_terms=p_terms, np_terms=np_terms, num_classes=n_classes, num_units=1024, learning_rate=0.00053)
model.fit(X_train, y_train, max_iter_ls=5, max_iter_backfitting=5)

# Predict on the test set
predicted_class, probs = model.predict(X_test)

# Print the results
print("Predicted classes:", predicted_class)
print("True classes:", y_test.tolist())
print("Predicted probabilities:\n", probs)

## Compute binary clasification metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_class, target_names=y.unique()))

# Compute the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_class)
print("Accuracy:", accuracy)