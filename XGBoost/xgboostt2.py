from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
data = pd.read_csv("XGBoost/diabetes.csv")

y = data["Outcome"]
X = data.drop(columns="Outcome",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

modelxgb = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=500, subsample=0.7, random_state=42)
modelxgb.fit(X_train,y_train)
tahminxgb = modelxgb.predict(X_test)

acsxgb = accuracy_score(y_test,tahminxgb)

print(acsxgb*100)

modelbay= GaussianNB()
modelbay.fit(X_train,y_train)
tahinbay = modelbay.predict(X_test)

acsbay = accuracy_score(y_test, tahinbay)
print(acsbay*100)

parametreler = {"max_depth":[3,5,7],
                "subsample":[0.2,0.5,0.7],
                "n_estimators":[500,1000,2000],
                "learning_rate":[0.2,0.5,0.7]}
grid = GridSearchCV(modelxgb,param_grid=parametreler,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
#'learning_rate': 0.7, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.7
print(grid.best_params_)

