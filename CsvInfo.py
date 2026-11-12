import pandas as pd
def printInfo(df:pd.DataFrame):
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.columns.tolist())