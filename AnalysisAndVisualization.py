import pandas as pd
import matplotlib.pyplot as plt
def AAV(df:pd.DataFrame):
    # 고객 이탈 현황
    churn_count = df["Churn"].value_counts()
    print(churn_count)
    churn_count.plot(kind="bar", title="Customer Churn")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.show()

    #고객 이탈 비율 확인
    churn_ratio = df["Churn"].value_counts(normalize=True) * 100
    print(churn_ratio)

    #가입 기간과 이탈 관계 분석
    plt.figure(figsize=(8,5))
    plt.hist(df[df["Churn"]=="Yes"]["tenure"],
             bins=20,
             alpha=0.6,
             label="Churn")
    plt.hist(df[df["Churn"]=="No"]["tenure"],
             bins=20,
             alpha=0.6,
             label="Stay")
    plt.xlabel("Tenure")
    plt.ylabel("Customer Count")
    plt.title("Churn by Tenure")
    plt.legend()
    plt.show()

    #월 이용 요금과 이탈 관계
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.title("MonthlyCharges by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.show()