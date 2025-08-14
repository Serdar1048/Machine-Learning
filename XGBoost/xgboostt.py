from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
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

modelxgb = XGBClassifier()
modelxgb.fit(X_train,y_train)
tahminxgb = modelxgb.predict(X_test)

acsxgb = accuracy_score(y_test,tahminxgb)

print(acsxgb*100)

modelbay= GaussianNB()
modelbay.fit(X_train,y_train)
tahinbay = modelbay.predict(X_test)

acsbay = accuracy_score(y_test, tahinbay)
print(acsbay*100)

