import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.title("App for SkLearn")

st.write('''
    ## Information about dataset
''')

dataset = st.sidebar.selectbox('Datasets', ['Iris', 'Wine', 'BreastCancer', 'Digits'])

model = st.sidebar.selectbox('Machine Learning Algorithm', ['SVM', 'KNN', 'GradientBoosting',
                                                            'Decision Tree', 'Random Forest'])


def load_data(dataset):
    if dataset == 'Iris':
        data = datasets.load_iris()
    elif dataset == 'Wine':
        data = datasets.load_wine()
    elif dataset == 'BreastCancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_digits()
    return data


data = load_data(dataset)
X = data.data
y = data.target
st.write(f"Dataset name : {dataset} dataset")
st.write(f"Shape of features: {X.shape}")
st.write(f"Shape of Targets: {y.shape}")


def create_param_sliders(model):
    params = dict()
    if model == 'SVM':
        C = st.sidebar.slider('C', 0.01, 1.0)
        params['C'] = C
    elif model == 'KNN':
        n_neighbors = st.sidebar.slider('K', 1, 15)
        params['n_neighbors'] = n_neighbors
    elif model == 'GradientBoosting':
        learning_rate = st.sidebar.slider('learning rate', 0.01, 10.00)
        n_estimators = st.sidebar.slider('n_estimators', 1, 1000)
        params['learning_rate'] = learning_rate
        params['n_estimators'] = n_estimators
    elif model == 'Decision Tree':
        max_depth = st.sidebar.slider('max_depth', 1, 100)
        params['max_depth'] = max_depth
    else:
        n_estimators = st.sidebar.slider('n_estimators', 1, 1000)
        max_depth = st.sidebar.slider('max_depth', 1, 100)
        params['n_estimators_rfc'] = n_estimators
        params['max_depth_rfc'] = max_depth
    return params


params = create_param_sliders(model)


def load_algorithm(model):
    if model == 'SVM':
        clf = SVC(C=params['C'])
    elif model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    elif model == 'GradientBoosting':
        clf = GradientBoostingClassifier(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
    elif model == 'Decision Tree':
        clf = DecisionTreeClassifier(max_depth=params['max_depth'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators_rfc'],max_depth=params['max_depth_rfc'])

    return clf


clf = load_algorithm(model)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {acc}")

# plotting
fig = plt.figure(figsize=(8, 3))
plt.scatter(X[:,0], X[:,1], c=y, cmap="viridis")
plt.xlabel("Principal componant 1")
plt.ylabel("Principal componant 2")
plt.colorbar()
st.pyplot(fig)

st.write(data.info())