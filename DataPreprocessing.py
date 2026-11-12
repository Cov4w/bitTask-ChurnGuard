import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def dataPreprocessing(df:pd.DataFrame):
    # 불필요한 컬럼 제거
    df = df.drop(columns=["customerID"])

    # 정답 데이터 생성
    X=df.drop(columns=["Churn"])
    y=df["Churn"].map({"No":0, "Yes":1})
    # 입력 데이터
    print(X.head())
    # 정답 데이터
    print(y.head())

    #수치형 및 범주형 컬럼 분리
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    # 수치형 컬럼
    print(numeric_features)
    # 범주형 컬럼
    print(categorical_features)

    # 훈련 데이터와 테스트 데이터 분리
    # stratify=y : 훈련 데이터와 테스트 데이터의 고객 이탈 비율을 비슷하게 유지하기 위한 설정
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("훈련 데이터 : ",X_train.shape)
    print("테스트 데이터 : ", y_train.shape)

    # 전처리 파이프라인 구성
    # 수치형 데이터 : 결측치를 중앙값으로 채우고 표준화
    # 범주형 데이터 : 최빈값으로 결측치를 채우고 원-핫 인코딩
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])
    print("데이터 전처리 완료")
    
    return preprocessor, X_train, X_test, y_train, y_test