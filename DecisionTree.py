import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def decisionTree(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, max_depth=None):
    tree_model = Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(max_depth=max_depth, random_state=42))])
    tree_model.fit(X_train, y_train)
    pred = tree_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    results = [accuracy, precision, recall, f1]
    return results, pred, tree_model