import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("iris.csv")

print("처음 5개 행:")
print(df.head())

print("\n행과 열의 개수:")
print(df.shape)

print("\n열 이름:")
print(df.columns.tolist())

# 입력: species를 제외한 네 개의 측정값
X = df.drop("species", axis=1)

# 정답: species 열
y = df["species"]

print("\n입력 X의 처음 5개 행:")
print(X.head())

print("\n정답 y의 처음 5개 값:")
print(y.head())

print("\nX의 크기:", X.shape)
print("y의 크기:", y.shape)

print("\n품종별 꽃의 개수:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n학습용 측정값:", X_train.shape)
print("시험용 측정값:", X_test.shape)
print("학습용 정답:", y_train.shape)
print("시험용 정답:", y_test.shape)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("\n결정 트리 학습 완료!")

y_pred = model.predict(X_test)

print("\n모델이 예측한 품종:")
print(y_pred)

comparison = pd.DataFrame({
    "실제 정답": y_test,
    "모델 예측": y_pred
})

print("\n정답과 예측 비교:")
print(comparison.head(10))

accuracy = accuracy_score(y_test, y_pred)

print("\n정확도:", accuracy)
print("맞힌 개수:", (y_test == y_pred).sum())
print("전체 시험 개수:", len(y_test))

print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

print("\n틀린 예측:")
print(comparison[comparison["실제 정답"] != comparison["모델 예측"]])


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

print("\n모델별 정확도:")

for name, clf in models.items():
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    score = accuracy_score(y_test, prediction)

    print(f"{name}: {score:.2%}")


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scaled_models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200)
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier()
    )
}

print("\n표준화 후 정확도:")

for name, clf in scaled_models.items():
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    score = accuracy_score(y_test, prediction)

    print(f"{name}: {score:.2%}")