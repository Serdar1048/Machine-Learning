import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("diyabetOrnek/diabetes.csv")
y = data["Outcome"]
X = data.drop(columns="Outcome",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def modeller(model):
    model.fit(X_train,y_train)
    tahmin = model.predict(X_test)
    skor = accuracy_score(y_test,tahmin)
    return round(skor*100,2)

models = []

models.append(("Log Regresyon", LogisticRegression(random_state=0)))
models.append(("KNN", KNeighborsClassifier()))
models.append(("SVC", SVC(random_state=0)))
models.append(("Bayes", GaussianNB()))
models.append(("Karar Ağacı", DecisionTreeClassifier(random_state=0)))

modelad = []
basari = []

for i in models:
    modelad.append(i[0])
    basari.append(modeller(i[1]))

a = list(zip(modelad,basari))
sonuc=pd.DataFrame(a,columns=["Model","Skor"])

print(sonuc)
