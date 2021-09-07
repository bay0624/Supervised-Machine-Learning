# Supervised-Machine-Learning
Building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.

# Background
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

The objective is to use this data to create machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier. Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier.

# Steps
## Load data source
```python
train_df = pd.read_csv(Path('Resources/2019loans.csv'))
test_df = pd.read_csv(Path('Resources/2020Q1loans.csv'))
```

## Convert categorical data to numeric and separate target feature for training and testing data
```python
X = train_df.drop('loan_status', axis=1)
X_train = pd.get_dummies(X)

X_1 = test_df.drop('loan_status', axis=1)
X_test = pd.get_dummies(X_1)
```

## Using LabelEncoder to convert output labels to binary (0 and 1)
```python
from sklearn.preprocessing import LabelEncoder
y_train = LabelEncoder().fit_transform(train_df['loan_status'])
y_test = LabelEncoder().fit_transform(test_df['loan_status'])
```

## Create Logistic Regression Model for the unscaled data
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
```
Training Data Score: 0.6485221674876848 <br>
Testing Data Score: 0.5253083794130158

### Create Confusion Matrix for the Logistic Regression Model for the unscaled data 
```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = y_test
y_pred = classifier.predict(X_test)
confusion_matrix(y_true, y_pred)
```

### Print out the Classification Report
```python
print(classification_report(y_true, y_pred))
```
<img src="https://github.com/bay0624/Supervised-Machine-Learning/blob/main/images/Class_Report1.png" width="600">

### Visualize the Confusion Matrix for the Logistic Regression Model
```python
import seaborn as sn
import matplotlib.pyplot as plt

confusion_df = pd.DataFrame(array)
sn.set(font_scale=1.4) # for label size
sn.heatmap(confusion_df, annot=True, annot_kws={"size": 14})
plt.show()
```
<img src="https://github.com/bay0624/Supervised-Machine-Learning/blob/main/images/Confusion_Matrix1.png" width="400">

## Repeat same steps as above for the scaled data
```python
# Scaling the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Training and Testing Score for scaled data (Logistic Regression Model)
Training Data Score: 0.713136288998358 <br>
Testing Data Score: 0.7201190982560612

### Classification Report for scaled data (Logistic Regression Model)
<img src="https://github.com/bay0624/Supervised-Machine-Learning/blob/main/images/Class_Report2.png" width="600">

### Confusion Matrix Heatmap for scaled data (Logistic Regression Model)
<img src="https://github.com/bay0624/Supervised-Machine-Learning/blob/main/images/Confusion_Matrix2.png" width="400">

## Create Random Forest Classifier for the unscaled data
```python
from sklearn.ensemble import RandomForestClassifier
clf_1 = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train, y_train)
print(f'Training Score: {clf_1.score(X_train, y_train)}')
print(f'Testing Score: {clf_1.score(X_test, y_test)}')
```
Training Data Score: 1.0 <br>
Testing Data Score: 0.6180348787749894

## Create Random Forest Classifier for the scaled data
```python
clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')
```
Training Data Score: 1.0 <br>
Testing Data Score: 0.6193109315185028

# Conclusion
### Logistic Regression Model
For the unscaled data, the scores for the Logistic Regression:
 - Training Data Score: 0.6485221674876848
 - Testing Data Score: 0.5253083794130158

For the scaled data, the scores for the Logistic Regression:
 - Training Data Score: 0.713136288998358
 - Testing Data Score: 0.7201190982560612

As one can see from the scores above, the best model to predict the Credit Risk is the Logistic Regression on the scaled data. The training scores and testing scores are much more closer than the unscaled data, hence allowing us to make a more accurate prediction.

### Random Forest Classifier
For unscaled data, the scores for the Random Forest Classifier Model:
 - Training Score: 1.0
 - Testing Score: 0.6180348787749894

For scaled data, the scores for the Random Forest Classifier Model:
 - Training Score: 1.0
 - Testing Score: 0.6193109315185028

As one can see from the scores above, the Random Forest Classifier won't be very efficient in predicting the Credit Risks. The differences between the scores on the scaled and unscaled data are almost identical (The difference is negligible. 

### In conclusion, for this exercise, Logistic Regression will be a better model to use.

