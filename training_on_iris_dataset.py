# training simple_perceptronwith iris dataset 
# setosa = 1, other = 0 (since we have to use binary data)

import numpy as np
from simple_perceptron import Perceptron

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the Iris dataset
# This dataset contains 150 samples, each with 4 features:
# sepal length, sepal width, petal length, and petal width.
iris = load_iris()

# Access the features and target variables
X = iris.data[:,:2] # selects sepal length and sepal width 
# X is an array with 150 rows and 2 clolumns
 
y = np.where(iris.target!=0, 0, 1)
# np.where(condition, value_if_true, value_if_false)
#           ['setosa' =0, 'versicolor'=1, 'virginica'=2]
# if iris.target == 0 then the function gives 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 2)
# we split the dataset into 20% for testing and 80% for training
# the randome_state = integer makes sure that we 'shuffle' the data first before we split 
# and that the 'shuffling' is consistent


# Create a perceptron with two input features
perceptron = Perceptron(learning_rate=0.1, epochs=10)

# Train the perceptron on the training data
perceptron.fit(X_train, y_train)

# Get predictions from the test set
y_pred = np.array([perceptron.predict(x) for x in X_test])


# Generate the classification report
report = classification_report(
  y_test, y_pred, 
  target_names=[
    'Iris-setosa',
    'Iris-other'
  ]
)

print(report)

""" understanding how classification_report() works...

Precision = True Positive / (True Positive + False POsitive)

Recall (Sensitivity) = True POsitive / (True Positive + False Negative)

F1-Score = harmonic mean of precision and recall
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Support = the number of true instances for each class in the dataset

Accuracy = The overall percentage of correct predictions across all classes.
Accuracy = (TP + TN) / (TP + TN + FP + FN)
"""

# getting a feel of the model's prediction
print(f"predicted    : {y_pred}")
print(f"correct label: {y_test}")

