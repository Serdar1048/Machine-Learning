import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
data = pd.read_csv("smsOrnek/spam.csv", encoding="Windows-1252")

data = data.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

data =data.rename(columns={"v1":"Etiket", "v2":"Sms"})

data = data.drop_duplicates()

data["Karakter Sayısı"] = data["Sms"].apply(len)

data.Etiket = [1 if kod=="spam" else 0 for kod in data.Etiket]
def harfler(cumle):
    yer = re.compile("[^a-zA-Z]")
    return re.sub(yer, " ", cumle)

spam = []
ham = []
tumcumleler = []
for i in range(len(data["Sms"].values)):
    r1 = data["Sms"].values[i]
    r2 = data["Etiket"].values[i]

    temizcumle = []
    cumleler = harfler(r1)
    cumleler = cumleler.lower()

    for kelimeler in cumleler.split():
        temizcumle.append(kelimeler)

        if r2 == 1:
            spam.append(cumleler)
        else:
            ham.append(cumleler)
    tumcumleler.append(" ".join(temizcumle))
data["Yeni Sms"] = tumcumleler

data = data.drop(columns=["Sms" ,"Karakter Sayısı"],axis=1)

cv = CountVectorizer()
x = cv.fit_transform(data["Yeni Sms"]).toarray()

y = data["Etiket"]
X = x

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

for i in np.arange(0.0,1.1,0.1):
    model = MultinomialNB(alpha=i)
    model.fit(X_train,y_train)
    tahmin = model.predict(X_test)
    skor = accuracy_score(y_test,tahmin)
    print("Alfa {} değeri için Skor: {}".format(round(i,1),round(skor*100,2)))
    
"""
durdurma = stopwords.words("english")
print(durdurma)
"""
"""
print(data["Sms"][0])
print(harfler(data["Sms"][0]))
"""
"""
data.hist(column="Karakter Sayısı", by="Etiket",bins=50)
plt.show()
"""