import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
data = pd.read_csv("naivebayes/kanser.csv")

data = data.drop(columns=["id", "Unnamed: 32"])

data.diagnosis = [1 if kod == "M" else 0 for kod in data.diagnosis]

y = data["diagnosis"]
X = data.drop(columns="diagnosis",axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

nbg = GaussianNB()
nbg.fit(X_train,y_train)
tahmin = nbg.predict(X_test)

acs = accuracy_score(y_test,tahmin)
print(acs*100)