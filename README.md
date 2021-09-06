# Supervised-Machine-Learning
Building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.

# Background
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

The objective is to use this data to create machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier.

# Steps
### Create Logistic Regression Model for unscaled data
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
```
Training Data Score: 0.6485221674876848 <br>
Testing Data Score: 0.5253083794130158

### Create Confusion Matrix for the Logistic Regression Model for unscaled data and printing out the Classification Report
```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = y_test
y_pred = classifier.predict(X_test)
array = confusion_matrix(y_true, y_pred)
array
```
