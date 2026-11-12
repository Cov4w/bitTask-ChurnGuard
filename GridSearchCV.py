import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

def gridSearchCV(rf_model:Pipeline, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    # 파라미터 후보 설정
    params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 7, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }
    grid = GridSearchCV(
        estimator=rf_model,
        param_grid=params,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("최적 파라미터")
    print(grid.best_params_)
    print("최고 교차 검증 Recall")
    print(grid.best_score_)
    best_model = grid.best_estimator_
    best_pred = best_model.predict(X_test)
    print("최적 모델 평가")
    print("Accuracy : ", accuracy_score(y_test, best_pred))
    print("Precision : ", precision_score(y_test, best_pred))
    print("Recall : ", recall_score(y_test, best_pred))
    print("F1 : ", f1_score(y_test, best_pred))    
    print("ROC-AUC : ", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))