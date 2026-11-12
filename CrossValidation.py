import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

def crossValidation(models:dict, X_train:pd.DataFrame, y_train:pd.DataFrame):
    cv_results = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=5, scoring=["accuracy", "precision", "recall", "f1"], n_jobs=-1)
        cv_results.append({"Model":name,
                           "Accuracy":scores["test_accuracy"].mean(),
                           "Precision":scores["test_precision"].mean(),
                           "Recall":scores["test_recall"].mean(),
                           "F1":scores["test_f1"].mean()
                           })
        cv_df = pd.DataFrame(cv_results)
        print(cv_df.round(4))
        cv_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", figsize=(12, 6))
        plt.title("5-Fold Cross Validation")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()