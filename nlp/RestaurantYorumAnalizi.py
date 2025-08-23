import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk.download("wordnet")
lema = nltk.WordNetLemmatizer()
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv("nlp/restaurant.tsv",delimiter="\t")
veri = data.copy()

temiz = []
for i in range(len(veri)):
    duzenle = re.sub('[^a-zA-Z]', ' ', veri["Review"][i])
    duzenle = duzenle.lower()
    duzenle = duzenle.split()
    duzenle = [lema.lemmatize(kelime) for kelime in duzenle if not kelime in set(stopwords.words("english"))]
    duzenle = ' '.join(duzenle)
    temiz.append(duzenle)

cv = CountVectorizer(max_features=1500)
matrix = cv.fit_transform(temiz).toarray()
#matrixdf = pd.DataFrame(matrix,columns=cv.get_feature_names_out())
#print(veri["Liked"].unique())

y = veri.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(matrix,y,test_size=0.2,random_state=42)

model = GaussianNB()
model.fit(X_train,y_train)
tahmin = model.predict(X_test)

skor = accuracy_score(y_test,tahmin)
print(skor*100)

model2 = RandomForestClassifier(random_state=0)
model2.fit(X_train,y_train)
tahmin2 = model2.predict(X_test)

skor2 = accuracy_score(y_test,tahmin2)
print(skor2*100)

#df = pd.DataFrame(list(zip(veri["Review"],temiz)),columns=["Orjinal Yorum","Temiz Yorum"])
#frekans = (df["Temiz Yorum"]).apply(lambda x:pd.value_counts(x.split())).sum(axis=0).reset_index()
#frekans.columns = ["Kelimeler","Frekans"]
#filtre = frekans[frekans["Frekans"]>10]
#filtre.plot.bar(x="Kelimeler",y="Frekans")
#plt.show()