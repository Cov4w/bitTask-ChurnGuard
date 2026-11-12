import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def SGDModel(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    sgd_model = Pipeline([("preprocessor", preprocessor),("model", SGDClassifier(loss="log_loss", max_iter=2000, random_state=42))])
    sgd_model.fit(X_train, y_train)

    # 예측 및 평가
    sgd_pred = sgd_model.predict(X_test)
    print("SGD Accuracy : ", accuracy_score(y_test, sgd_pred))
    print("SGD Precision : ", precision_score(y_test, sgd_pred))
    print("SGD Recall : ", recall_score(y_test, sgd_pred))
    print("SGD F1 : ", f1_score(y_test, sgd_pred))
    result = classification_report(y_test, sgd_pred)
    return result