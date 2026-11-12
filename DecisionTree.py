import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def decisionTree(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    tree_model = Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(random_state=42))])
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    print("Decision Tree Accuracy : ", accuracy_score(y_test, tree_pred))
    print("Decision Tree Recall : ", recall_score(y_test, tree_pred))
    print("Decision Tree F1 : ", f1_score(y_test, tree_pred))

    # 트리 깊이에 따른 성능 비교
    depths = [2, 3, 5, 7, 10, None]
    for depth in depths:
        model = Pipeline([("preprocessor", preprocessor), ("model",DecisionTreeClassifier(max_depth=depth, random_state=42))])
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Depth = {depth}", f"Train={train_score : .4f}", f"Test = {test_score : .4f}")
    