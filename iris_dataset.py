from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
target =iris.target

# Print the first 5 rows of the feature data and labels
# Feature names:
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Target names:
# ['setosa' =0, 'versicolor'=1, 'virginica'=2]

print("First 5 rows of the feature data (X):")
print(iris.data[:5])
print("First 5 labels:", iris.target[:5])  


