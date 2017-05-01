from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loads iris data
iris_dataset = load_iris()

#prints all the keys in the data
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

#splits dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#Data visualization
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

#Model building using k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=1)

#fitting the model
knn.fit(X_train, y_train)

#Evaluating the model
accuracy = knn.score(X_test, y_test)
print("The accuracy of the model is " + str(accuracy) )