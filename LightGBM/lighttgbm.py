import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("lightGBM/diabetes.csv")
print(data)

y = data["Outcome"]
X = data.drop(columns="Outcome",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LGBMClassifier()
model.fit(X_train,y_train)
tahmin = model.predict(X_test)

acs = accuracy_score(y_test,tahmin)
print(acs*100)

parametreler = param_grid = {
    "learning_rate": [0.1],
    "n_estimators": [100, 300],
    "subsample": [0.6,0.8],
    "max_depth": [3, 5]
}
grid = GridSearchCV(model,param_grid=parametreler,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)