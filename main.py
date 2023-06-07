# Cancellation behavior analysis holds immense value for Agoda. By leveraging machine learning, we
# can revolutionize our ability to make quick decisions and provide our partners, who may not have
# access to extensive booking records, with invaluable insights into cancellation patterns. Your task is
# to harness the power of machine learning to create predictive models that forecast whether a guest
# will stay at a hotel or cancel their booking. But it doesn’t stop there! We also need to determine the
# most likely timing of cancellations.
# Your solution will empower Agoda to optimize the net room nights at the booking level, resulting in
# improved decision-making, enhanced efficiency, and, ultimately, happier guests. So, gear up, get
# creative, and let your passion for machine learning transform how we understand and manage hotel
# cancellations.
# Dataset
# The training dataset Agoda_training.csv holds 58,659 records with 38 features determined upon
# booking and the unknown variable of interest, the “cancellation_datetime” which features the date of
# cancellation when it is such, see Table 1. To evaluate your predictions, we test you on 7,818 booking
# records, split into two equal size files, for which you are given only part of the booking information -
# the missing data is specified in each task.
# The test data sets Agoda_Test_1.csv and Agoda_Test_2.csv correspond to tasks 1.2.1 and 1.2.2
# Tasks
# Please note that the tasks are independent. Partial submission will grant you a partial grade. Please
# read the submission guides carefully - if your output doesn’t exactly match these definitions it will
# regarded as no submission.
# Cancellation prediction
# Given the booking information, we would like to predict whether this order will or will not be
# canceled.
# The input A dataframe of the booking information – all columns except “cancellation_datetime”.
# The output A csv file named “agoda_cancellation_prediction.csv” with two columns:- id (h_booking_id)
# and cancellation: where 1 indicating that a cancellation is predicted, and 0 otherwise.
# Evaluation We will evaluate your predictions according to their F1 macro metric. An example of
# the output:
# ID cancellation
# 111 1
# 222 0
# 333 1
# Table 2: Cancellation prediction output
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tools.preprocess import Preprocess

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_dates(data):
    data['booking_datetime'] = pd.to_datetime(data['booking_datetime'])
    data['checkin_date'] = pd.to_datetime(data['checkin_date'])
    data['checkout_date'] = pd.to_datetime(data['checkout_date'])
    # remove hours:
    data['checkin_date'] = data['checkin_date'].dt.date
    data['checkout_date'] = data['checkout_date'].dt.date
    data['booking_datetime'] = data['booking_datetime'].dt.date

    # add column of how much time passed between booking and checkin and name it
    data['booking_checkin_diff'] = data['checkin_date'] - data['booking_datetime']
    # add column of how much time passed between checkin and checkout
    data['checkin_checkout_diff'] = data['checkout_date'] - data['checkin_date']
    return data
def preprocess_data(data):
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == type(object):
            data[column] = le.fit_transform(data[column].astype(str))
    return data

def drop_ids(data):
    data = data.drop(["h_booking_id", "h_customer_id"], axis=1)
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def ridge_regression(X_train, y_train, alpha=0.001):
    model = LogisticRegression(penalty='l2', C=1/alpha, intercept_scaling=1000)
    model.fit(X_train, y_train)
    return model



def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, kernel='rbf'):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print("F1 Score:", f1)

def preprocess_test_data(test_data, label_encoder):
    for column in test_data.columns:
        if test_data[column].dtype == type(object):
            test_data[column] = label_encoder.transform(test_data[column].astype(str))
    return test_data

def generate_predictions(model, test_data, id_column):
    test_predictions = model.predict(test_data)
    submission = pd.DataFrame({
        'ID': test_data[id_column],
        'cancellation': test_predictions
    })
    return submission

def save_submission(submission, file_path):
    submission.to_csv(file_path, index=False)


if __name__ == "__main__":
    data = load_data(r"data\agoda_cancellation_train.csv")
    # add a column with 1 0 if there os cancelation datetime or not
    # print amount of nans in cancellation_datetime
    data["is_cancelled"] = data["cancellation_datetime"].apply(lambda x: 0 if pd.isna(x) else 1)
    data = data.drop(["cancellation_datetime"], axis=1)
    data = data.dropna()
    # check if there is NoneType in the data
    print(data.isnull().sum())
    data = Preprocess._preprocess_numericals(data)
    data = Preprocess.process_dates(data)
    X_train, X_test, y_train, y_test = split_data(data, "is_cancelled")
    # preprocess data
    linear_regression_model = linear_regression(X_train, y_train)
    ridge_regression_model = ridge_regression(X_train, y_train)
    random_forest = train_random_forest(X_train, y_train)
    knn = train_knn(X_train, y_train)
    svm = train_svm(X_train, y_train)
    decision_tree = train_decision_tree(X_train, y_train)
    

    # evaluate models and plot confusion matrix
    models = [linear_regression_model, ridge_regression_model, random_forest, knn, svm, decision_tree]
    models_eval = []
    for model in models:
        print("Model:", model)
        evaluate_model(model, X_test, y_test)
        models_eval.append(model)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, model.predict(X_test)))
        print("Accuracy score:", accuracy_score(y_test, model.predict(X_test)))
        print("Precision score:", precision_score(y_test, model.predict(X_test)))
        print("Recall score:", recall_score(y_test, model.predict(X_test)))
        print("F1 score:", f1_score(y_test, model.predict(X_test)))
        print("--------------------------------------------------")



    