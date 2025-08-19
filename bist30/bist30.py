import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx?endeks=03#page-1"

r = requests.get(url)

s = BeautifulSoup(r.text,"html.parser")

tablo = s.find("table", {"id":"summaryBasicData"})
tablo = pd.read_html(str(tablo),flavor="bs4")[0]
hisseler = []

for i in tablo["Kod"]:
    hisseler.append(i)

hisseler = [h for h in hisseler if h != "ASTOR"]    

parametreler = (
    ("hisse",hisseler[0]),
    ("startdate","28-11-2020"),
    ("enddate","28-11-2022"))    

url2 = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
r2 = requests.get(url2,params=parametreler).json()["value"]
veri = pd.DataFrame.from_dict(r2)
veri = veri.iloc[:,0:3]
veri = veri.rename({"HGDG_HS_KODU":"Hisse","HGDG_TARIH":"Tarih","HGDG_KAPANIS":"Fiyat"},axis=1)

data = {"Tarih":veri["Tarih"],veri["Hisse"][0]:veri["Fiyat"]}
veri = pd.DataFrame(data)

del hisseler[0]
tumveri = [veri]

for j in hisseler:
    parametreler = (
        ("hisse",j),
        ("startdate","28-11-2020"),
        ("enddate","28-11-2022"))    

    url2 = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
    r2 = requests.get(url2,params=parametreler).json()["value"]
    veri = pd.DataFrame.from_dict(r2)
    veri = veri.iloc[:,0:3]
    veri = veri.rename({"HGDG_HS_KODU":"Hisse","HGDG_TARIH":"Tarih","HGDG_KAPANIS":"Fiyat"},axis=1)

    data = {"Tarih":veri["Tarih"],veri["Hisse"][0]:veri["Fiyat"]}
    veri = pd.DataFrame(data)
    tumveri.append(veri)

df = tumveri[0]

for son in tumveri[1:]:
    df = df.merge(son,on="Tarih")

veri = df.drop(columns="Tarih",axis=1)

gelir = veri.pct_change().mean()*252

sonuc = pd.DataFrame(gelir)
sonuc.columns = ["Gelir"]

sonuc["Oynaklık"] = veri.pct_change().std() * np.sqrt(252)
sonuc = sonuc.reset_index()
sonuc = sonuc.rename({"index":"Hisse"},axis=1)
sonuc.to_csv("bist30/veri.csv")
print(sonuc)