import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

def ModelPC(logistic_model:Pipeline, sgd_model:Pipeline, tree_model:Pipeline, rf_model:Pipeline, gb_model:Pipeline, 
            X_test: pd.DataFrame, y_test: pd.Series):
    models = {"Logistic Regression": logistic_model,
        "SGD Classifier": sgd_model,
        "Decision Tree": tree_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model
        }

    def evaluate_model(name, model):
        pred = model.predict(X_test)
        return {
            "Model":name,
            "Accuracy":accuracy_score(y_test, pred),
            "Precision":precision_score(y_test, pred),
            "Recall":recall_score(y_test, pred),
            "F1":f1_score(y_test, pred)
        }
    results = []
    for name, model in models.items():
        result = evaluate_model(name, model)
        results.append(result)

    result_df = pd.DataFrame(results)
    print("\n[모델별 성능 비교 결과]")
    print(result_df.round(4))
    result_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Machine Learning Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return models, result_df