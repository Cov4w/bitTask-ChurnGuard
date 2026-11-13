import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def SGDModel(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, alpha_value=0.0001):
    sgd_model = Pipeline([("preprocessor", preprocessor),("model", SGDClassifier(alpha=alpha_value, loss="log_loss", max_iter=2000, random_state=42))])
    sgd_model.fit(X_train, y_train)

    # 예측 및 평가
    pred = sgd_model.predict(X_test)
   
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    result = classification_report(y_test, pred)
    return [accuracy, precision, recall, f1]