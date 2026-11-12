import pandas as pd

def readCsv():
    url = (
        "https://raw.githubusercontent.com/"
        "IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],    errors="coerce")
    return df
