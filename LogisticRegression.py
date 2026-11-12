import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

def logisticRegression(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    # 모델 생성
    logistic_model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    # 모델 학습
    logistic_model.fit(X_train, y_train)
    print("로지스틱 회귀 학습 완료")

    # 예측
    pred = logistic_model.predict(X_test)
    print("예측 결과 : ", pred[:20])
    print("실제 결과 : ", y_test.values[:20])
    # 모델 평가
    print("Accuracy : ", accuracy_score(y_test, pred))
    print("Precision : ", precision_score(y_test, pred))
    print("Recall : ", recall_score(y_test, pred))
    print("F1-score : ", f1_score(y_test, pred))
    print("Log Loss : ", log_loss(y_test, pred))
    result = classification_report(y_test, pred)
    return result