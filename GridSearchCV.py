import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

def gridSearchCV(models_dict: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    
    # 1. 각 모델별 하이퍼파라미터 탐색 범위 설정
    param_grids = {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
        "SGD Classifier": {"model__alpha": [0.0001, 0.001, 0.01]},
        "Decision Tree": {"model__max_depth": [3, 5, 7, 10]},
        "Random Forest": {"model__n_estimators": [100, 200, 300]},
        "Gradient Boosting": {"model__learning_rate": [0.01, 0.1, 0.2]}
    }
    
    report_list = []
    best_overall_model_name = ""
    best_overall_f1 = -1.0
    best_overall_params = {}
    best_overall_metrics = {}

    # 2. 모아둔 5개 모델 딕셔너리를 순회하며 한 번에 GridSearchCV 수행
    for model_name, pipeline_model in models_dict.items():
        grid = GridSearchCV(
            estimator=pipeline_model,
            param_grid=param_grids[model_name],
            cv=5,                 # 5-Fold 교차 검증
            scoring="f1",         # 불균형 데이터이므로 F1 점수를 기준으로 최적 모델 선정
            n_jobs=-1             # 모든 CPU 코어를 사용하여 속도 향상
        )
        
        # 학습 데이터로 그리드 서치 수행
        grid.fit(X_train, y_train)
        
        # 최적의 하이퍼파라미터가 적용된 모델(best_estimator_)로 테스트 데이터 예측
        best_model = grid.best_estimator_
        pred = best_model.predict(X_test)
        
        # [추가] ROC-AUC 계산 (SGDClassifier처럼 predict_proba가 없는 모델은 decision_function 사용)
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_test)[:, 1]
        else:
            y_score = best_model.decision_function(X_test)
        roc_auc = round(float(roc_auc_score(y_test, y_score)), 4)

        # 테스트 세트 기본 성능 4가지 계산
        acc = round(float(accuracy_score(y_test, pred)), 4)
        prec = round(float(precision_score(y_test, pred)), 4)
        rec = round(float(recall_score(y_test, pred)), 4)
        f1 = round(float(f1_score(y_test, pred)), 4)
        
        # 'model__C': 1 형태에서 'model__' 접두사를 떼어내고 깔끔하게 정리
        clean_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
        
        # [핵심 수정] WPF DataGrid의 '모델' 열에 최적 파라미터가 함께 출력되도록 문자열 생성
        # 예: "Logistic Regression (C=1)"
        param_str = ", ".join([f"{k}={v}" for k, v in clean_params.items()])

        # 모델별 탐색 결과 저장 (C#의 ModelMetric 클래스 프로퍼티와 매칭됨)
        model_info = {
            "ModelName": f"{model_name} ({param_str})", # DataGrid 첫 번째 열에 표시될 이름
            "BestParams": str(clean_params),
            "CV_BestScore": round(float(grid.best_score_), 4),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
        report_list.append(model_info)

        # 3. 5개 모델 중 가장 성능(F1 기준)이 높은 최종 우승 모델 갱신
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model_name = model_name
            best_overall_params = clean_params
            best_overall_metrics = model_info

    # 4. 최종 모델 선정 보고서 텍스트 작성 (콘솔 출력 및 C# WPF 전송용)
    final_report_text = (
        f"========== [최종 모델 선정 보고서] ==========\n"
        f"1. 최종 선정 모델 : {best_overall_model_name}\n"
        f"2. 최적 하이퍼파라미터 : {best_overall_params}\n"
        f"3. 교차검증(CV) 최고 점수 : {best_overall_metrics['CV_BestScore']}\n"
        f"4. 테스트 데이터 최종 성능 :\n"
        f"   - Accuracy  : {best_overall_metrics['Accuracy']}\n"
        f"   - Precision : {best_overall_metrics['Precision']}\n"
        f"   - Recall    : {best_overall_metrics['Recall']}\n"
        f"   - F1 Score  : {best_overall_metrics['F1']}\n"
        f"   - ROC-AUC   : {best_overall_metrics['ROC_AUC']}\n"
        f"5. 선정 사유 : 5개 후보 모델에 대해 5-Fold GridSearchCV를 수행한 결과, "
        f"{best_overall_model_name} 모델이 최적 파라미터({best_overall_params})에서 "
        f"테스트 F1 점수 {best_overall_f1}(ROC-AUC: {best_overall_metrics['ROC_AUC']})로 "
        f"가장 우수한 고객 이탈 예측 성능을 보였으므로 최종 모델로 선정함.\n"
        f"============================================="
    )

    print(final_report_text)

    # C# WPF에서도 표와 텍스트로 동시에 쓸 수 있도록 딕셔너리 형태로 반환
    return {
        "SummaryReport": final_report_text,      # 최종 보고서 줄글 텍스트
        "BestModel": best_overall_metrics,       # 최종 선정된 1등 모델 정보
        "AllGridResults": report_list            # 5개 모델 전체 GridSearch 결과 리스트 (DataGrid 바인딩용)
    }