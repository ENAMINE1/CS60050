# Group Number: 42
# Roll Numbers: 21ME30078 Debraj Das
#               21CS30047 Shashwat Kumar
# Project Number: NANB
# Project Title: Nursery School Application Selection using Naive Bayes Algorithm

# Importing necessary libraries
import numpy as np
import pandas as pd
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
# for comparison with scikit-learn's Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB

# Loading the dataset from a CSV file
data = pd.read_csv("nursery.csv")
original_data = data.copy()

# Converting categorical variables to numerical using label encoding
for column in data.columns:
    data[column] = pd.factorize(data[column])[0]

# print(data.head())

# Separating features and target variable
X = data.drop("final evaluation", axis=1)
y = data["final evaluation"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# Implementing a Naive Bayes Classifier from scratch
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_probabilities = None
        self.feature_probabilities = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        # print("Classes:", self.classes)
        # Calculating class probabilities
        self.class_probabilities = {}
        for c in self.classes:
            self.class_probabilities[c] = np.sum(y == c) / n_samples

        # Calculating feature probabilities
        self.feature_probabilities = defaultdict(dict)
        for feature in range(n_features):
            # print("Original Categorical Names:")
            # print(original_data.columns[feature])
            # print(original_data[original_data.columns[feature]].unique())
            for c in self.classes:
                # print("Feature:", feature, "Class:", c)
                # print original categorical values
                feature_values = X[y == c, feature]
                unique_values = np.unique(feature_values)
                for v in unique_values:
                    self.feature_probabilities[c][feature, v] = np.sum(
                        feature_values == v
                    ) / np.sum(y == c)
                    # print("Feature:", feature, "Class:", c, "Value:", v, "Probability:", self.feature_probabilities[c][feature, v])

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {
                c: log(self.class_probabilities[c]) for c in self.classes
            }
            for c in self.classes:
                for feature, value in enumerate(x):
                    if (feature, value) in self.feature_probabilities[c]:
                        class_scores[c] += log(self.feature_probabilities[c][feature, value])
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions

# Training our Naive Bayes Classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train.values, y_train.values)

# Predicting on the test set
y_pred_custom = nb_classifier.predict(X_test.values)

# Training scikit-learn's Naive Bayes Classifier
sklearn_nb_classifier = GaussianNB()
sklearn_nb_classifier.fit(X_train, y_train)
y_pred_sklearn = sklearn_nb_classifier.predict(X_test)

# Comparing the results
print("Custom Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_custom))
print("Classification Report:")
print(classification_report(y_test, y_pred_custom, zero_division=0))

print("\nScikit-learn's Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_sklearn))
print("Classification Report:")
print(classification_report(y_test, y_pred_sklearn, zero_division=0))
