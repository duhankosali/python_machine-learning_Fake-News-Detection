# Random Forest Algoritması ile Makine Öğrenmesi Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer kullanmak için

from merge_database import mergeDatabase

df_connect = mergeDatabase()
print(df_connect)

x = df_connect["text"]
y = df_connect["status"]

print("Result: ")
print(x)

from sklearn.ensemble import RandomForestClassifier # Random Forest Kütüphanesi
from sklearn.feature_extraction.text import TfidfVectorizer # Vektörize etme kütüphanesi

import pickle # Scriptleri kaydetmek için kullanılır

# Veri setimizi eğitim, doğrulama ve test veri setlerine ayırıyoruz
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2)

# İşlediğimiz verileri vektörel hale getiriyoruz.
vectorize = CountVectorizer() # vectorize nesnesi oluşturuyoruz
xv_train = vectorize.fit_transform(x_train) # Eğitim verimizi vektörize ediyoruz.
xv_test = vectorize.transform(x_test) # Test verimizi vektörize ediyoruz.
xv_val = vectorize.transform(x_val) # Doğrulama verimizi vektörize ediyoruz.
# Eğitilmiş vektörü kaydedin
with open('Vectorizer/CountVectorizer.pkl', 'wb') as f:
    pickle.dump(vectorize, f)

# Random Forest algoritması ile test-train yapıyoruz ve %98 lik bir başarı elde ettik.
# Yukarda kütüphaneyi import ettik.
rfc = RandomForestClassifier(random_state=0) # Random Forest sınıfı oluşturduk.
rfc.fit(xv_train, y_train) # Eğitim verisi olarak ayırdığımız Vektörel metinlerimiz ve 0,1 değerlerimiz ile eğitim gerçekleştirdik. (Verilerimizin %75'i eğitim verisidir)
print("Skor %", rfc.score(xv_test, y_test)*100) # Test verisi olarak ayırdığımız Vektörel metinlerimiz ve 0,1 değerlerimiz ile %98 lik bir başarı skorumuz var.
# modeli kaydedin
# with open('RandomForest/rfcModal.pkl', 'wb') as f:
#     pickle.dump(rfc, f)