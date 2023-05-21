# Random Forest Algoritması ile Makine Öğrenmesi Script

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer kullanmak için
from sklearn.feature_extraction.text import TfidfVectorizer # Vektörize etme kütüphanesi

from merge_database import mergeDatabase # Fake.csv ve Real.csv veritabanlarımı birleştiriyor.

df_connect = mergeDatabase()
print(df_connect)

x = df_connect["text"]
y = df_connect["status"]

from sklearn.ensemble import RandomForestClassifier # Random Forest Kütüphanesi

import pickle # Scriptleri kaydetmek için kullanılır

# Veri setimizi eğitim, doğrulama ve test veri setlerine ayırıyoruz
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5)

# İşlediğimiz verileri vektörel hale getiriyoruz.
# vectorize = CountVectorizer() # vectorize nesnesi oluşturuyoruz
vectorize = TfidfVectorizer() # vectorize nesnesi oluşturuyoruz
xv_train = vectorize.fit_transform(x_train) # Eğitim verimizi vektörize ediyoruz.
xv_test = vectorize.transform(x_test) # Test verimizi vektörize ediyoruz.
xv_val = vectorize.transform(x_val) # Doğrulama verimizi vektörize ediyoruz.
# Eğitilmiş vektörü kaydedin
# with open('Vectorizer/TfidfVectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorize, f)

# Random Forest algoritması ile test-train yapıyoruz ve %98 lik bir başarı elde ettik.
# Yukarda kütüphaneyi import ettik.
n_estimators_options = [50, 100, 200, 300]
max_depth_options = [10, 20, 30, 40, 50]

best_val_score = 0
best_n_estimators = None
best_max_depth = None

for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        rfc.fit(xv_train, y_train)
        val_score = rfc.score(xv_val, y_val)
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_n_estimators = n_estimators
            best_max_depth = max_depth

print(f"En iyi doğrulama skoru: {best_val_score}")
print(f"En iyi n_estimators: {best_n_estimators}")
print(f"En iyi max_depth: {best_max_depth}")

# En iyi hiperparametrelerle modeli yeniden eğitin
rfc_best = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=0)
rfc_best.fit(xv_train, y_train)

# Test veri setinde modelin performansını değerlendirin
test_score = rfc_best.score(xv_test, y_test)
print(f"Test skoru: {test_score}")

# Modeli kaydedin
# with open('RandomForest/rfcModal_best.pkl', 'wb') as f:
#     pickle.dump(rfc_best, f)

# # Notlar:
# "rfc_TfidfVectorizer.py" isimli dosyada veri setini yalnızca eğitim ve test olarak ayırmıştım. Fakat bu projede eğitim, test ve doğrulama olarak 3'e ayırdım. 
# Doğrulama seti, modelinizi ayarlamanız ve hiperparametrelerini seçmeniz için kullanılır.
# Ben burada "n_estimators" ve "max_depth" hiperparametrelerini kullandım. 
# Aldığım sonuç: En iyi n_estimators = 300, En iyi max_depth = 40
# Proje raporunda Random Forest'in ne olduğu TfidfVectorizer'in ne olduğu ve ayriyeten kullandığım hiperparametrelerin ne olduğu hakkında detaylı bilgi yer almalı.
