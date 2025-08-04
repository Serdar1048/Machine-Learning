import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# diagnosis: M-> kötü huylu tümör    B-> iyi huylu tümör
data = pd.read_csv("logistic-reg/kanser.csv")

data = data.drop(columns=["id","Unnamed: 32"],axis=1)

y = data["diagnosis"]
X = data.drop(columns="diagnosis",axis=1)

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression(random_state=0)
model.fit(X_train,y_train)
tahmin = model.predict(X_test)

print(tahmin)




