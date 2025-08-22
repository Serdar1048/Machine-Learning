import pandas as pd
from pyECLAT import ECLAT

data = pd.read_csv("eclat/bakkal.csv",header=None)

veri = data.copy()
veri.columns = ["Ürün"]
veri = list(veri["Ürün"].apply(lambda x:x.split(",")))

veri2 = pd.DataFrame(veri)
minürün = 2
mindestek = 0.2
maxürün = max([len(x) for x in veri ])

ec = ECLAT(veri2,verbose=True)
a,b = ec.fit(min_support=mindestek,min_combination=minürün,max_combination=maxürün)
print(b)