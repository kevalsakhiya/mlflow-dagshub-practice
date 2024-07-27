# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import dagshub

dagshub.init(repo_owner='kevalsakhiya', repo_name='mlflow-dagshub-practice', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/kevalsakhiya/mlflow-dagshub-practice.mlflow')



# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree model
mlflow.set_experiment('iris-decision-tree')
mlflow.autolog()

with mlflow.start_run():
    max_depth = 5
    random_state = 42
    clf = DecisionTreeClassifier(max_depth=max_depth,random_state=random_state)
    clf.fit(X_train, y_train)

    # Predict the test set results
    y_pred = clf.predict(X_test)

    # Calculate classification metrics
    print("Classification Report:")
    classi_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    # mlflow.log_metric('accuracy',accuracy)
    # mlflow.log_metric('precision',precision)
    # mlflow.log_metric('recall',recall)
    # mlflow.log_metric('f1_score',f1)
    # mlflow.log_metric('classification-report',classi_report)


    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # artifect
    # mlflow.log_artifact('confusion_matrix.png')
    # mlflow.log_artifact(__file__)
    # mlflow.sklearn.log_model(clf,'Decision Tree')

    # # tag
    # mlflow.set_tag('Author','Keval')
    # mlflow.set_tag('Model','DecisionTree')

    print('Finished')
