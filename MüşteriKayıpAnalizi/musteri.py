import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("MüşteriKayıpAnalizi/musteri.csv")
data = data.drop(columns="customerID",axis=1)

data = data.rename(columns={"gender":"Cinsiyet",
                    "SeniorCitizen":"65 yaş üstü",
                    "Partner":"Medeni Durum",
                    "Dependents":"Bakma Sorumluluğu",
                    "tenure":"Müşteri Olma Süresi (Ay)",
                    "PhoneService":"Ev Telefonu Aboneliği",
                    "MultipleLines":"Birden Fazla Abonelik Durumu",
                    "InternetService":"İnternet Aboneliği",
                    "OnlineSecurity":"Güvenlik Hizmeti Aboneliği",
                    "OnlineBackup":"Yedekleme Hizmeti Aboneliği",
                    "DeviceProtection":"Ekipman Güvenlik Aboneliği",
                    "TechSupport":"Teknik Destek Aboneliği",
                    "StreamingTV":"IP Tv Aboneliği",
                    "StreamingMovies":"Film Aboneliği",
                    "Contract":"Sözleşme Süresi",
                    "PaperlessBilling":"Online Fatura (Kağitsiz)",
                    "PaymentMethod":"Ödeme Şekli",
                    "MonthlyCharges":"Aylik Ücret",
                    "TotalCharges":"Toplam Ücret",
                    "Churn":"Kayip Durumu"})

data["Cinsiyet"] = ["Erkek" if kod == "Male" else "Kadın" for kod in data["Cinsiyet"]]
data["65 yaş üstü"] = ["Evet" if kod == 1 else "Hayır" for kod in data["65 yaş üstü"]]
data["Medeni Durum"] = ["Evli" if kod == "Yes" else "Bekar" for kod in data["Medeni Durum"]]
data["Bakma Sorumluluğu"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Bakma Sorumluluğu"]]
data["Ev Telefonu Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Ev Telefonu Aboneliği"]]
data["Birden Fazla Abonelik Durumu"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Birden Fazla Abonelik Durumu"]]
data["İnternet Aboneliği"] = ["Yok" if kod == "No" else "Var" for kod in data["İnternet Aboneliği"]]
data["Güvenlik Hizmeti Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Güvenlik Hizmeti Aboneliği"]]
data["Yedekleme Hizmeti Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Yedekleme Hizmeti Aboneliği"]]
data["Ekipman Güvenlik Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Ekipman Güvenlik Aboneliği"]]
data["Teknik Destek Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Teknik Destek Aboneliği"]]
data["IP Tv Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["IP Tv Aboneliği"]]
data["Film Aboneliği"] = ["Var" if kod == "Yes" else "Yok" for kod in data["Film Aboneliği"]]
data["Sözleşme Süresi"] = ["1 Aylık" if kod == "Month-to-month" else "1 Yıllık" if kod == "One year" else "2 Yıllık" for kod in data["Sözleşme Süresi"]]
data["Online Fatura (Kağitsiz)"] = ["Evet" if kod == "Yes" else "Hayır" for kod in data["Online Fatura (Kağitsiz)"]]
data["Ödeme Şekli"] = ["Elektronik" if kod == "Electronic check" else "Mail" if kod == "Mailed check" else "Havale" if kod == "Bank transfer (automatic)" else "Kredi Kartı" for kod in data["Ödeme Şekli"]]
data["Kayip Durumu"] = ["Evet" if kod == "Yes" else "Hayır" for kod in data["Kayip Durumu"]]

data["Toplam Ücret"] = pd.to_numeric(data["Toplam Ücret"],errors="coerce")
#print(data.info())

#for i in data.columns: 
#    print(i)
#    print(data[i].unique())

data["Fark"] = data["Toplam Ücret"]-(data["Müşteri Olma Süresi (Ay)"] * data["Aylik Ücret"] )
#print(data[data["Müşteri Olma Süresi (Ay)"]==0])
data = data.dropna()

#plt.boxplot(data["Müşteri Olma Süresi (Ay)"])
#plt.show()

le = LabelEncoder()
degisken = data.select_dtypes(include="object").columns 
data.update(data[degisken].apply(le.fit_transform))                                                             
data["Kayip Durumu"] = [1 if kod == 0 else 0 for kod in data["Kayip Durumu"]]

y = data["Kayip Durumu"]
X = data.drop(columns="Kayip Durumu",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = LazyClassifier()
modeller, tahmin = clf.fit(X_train, X_test, y_train, y_test)
sira = modeller.sort_values(by="Accuracy",ascending=True)
plt.barh(sira.index,sira["Accuracy"])
plt.show()

models = ["LinearSVC","SVC","Ridge","Logistic","RandomForest","LGBM","XGBM"]
sınıflar = [LinearSVC(random_state=0),SVC(random_state=0),
            RidgeClassifier(random_state=0),
            LogisticRegression(random_state=0),
            RandomForestClassifier(random_state=0),
            LGBMClassifier(random_state=0),XGBClassifier()]

parametreler = {
    models[0]:{"C":[0.1,1,10,100],"penalty":["l1","l2"]},
    models[1]:{"kernel":["linear","rbf"],"C":[0.1,1],"gamma":[0.01,0.001]},
    models[2]:{"alpha":[0.1,1.0]},
    models[3]:{
    "C": [0.1, 1],
    "penalty": ["l2"],  # sadece l2 kullan
    "solver": ["lbfgs"]},
    models[4]:{"n_estimators":[1000,2000],"max_depth":[4,10],"min_samples_split":[2,5]},
    models[5]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10],"subsample":[0.6,0.8]},
    models[6]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10],"subsample":[0.6,0.8]}
}

def cozum(model):
    model.fit(X_train,y_train)
    return model

def skor(model2):
    tahmin = cozum(model2).predict(X_test)
    acs = accuracy_score(y_test,tahmin)
    return acs*100

#m=LinearSVC(random_state=0)
#m.fit(X_train,y_train)
#t = m.predict(X_test)
#s=accuracy_score(y_test,t)
#print(s*100)

#print(skor(sınıflar[0]))

for i, j in zip(models,sınıflar):
    print(i)
    grid = GridSearchCV(cozum(j),param_grid=parametreler[i],cv=10,n_jobs=-1)
    grid.fit(X_train,y_train)
    print(grid.best_params_)

basarı = []
for i in sınıflar:
    basarı.append(skor(i))

print(basarı)

a = list(zip(models,basarı))
sonuc = pd.DataFrame(a,columns=["Model","Başarı"])
print(sonuc.sort_values("Başarı",ascending=False))