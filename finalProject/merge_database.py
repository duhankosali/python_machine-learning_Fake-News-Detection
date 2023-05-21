import pandas as pd
from sentence_processing import sentenceProcessing

# Veri setlerimizi tanımlıyoruz:
fake = pd.read_csv("verisetleri/fake_news.csv") 
real = pd.read_csv("verisetleri/real_news.csv")

fake["status"] = 0 # Sahte haberler için "status" == 0
real["status"] = 1 # Gerçek haberler için "status" == 1

print(fake.head(5))
print(real.head(5))

# Yalnızca "text" ve "status" verilerine ihtiyacımız var bu nedenle geriye kalanları siliyoruz.
df_fake=fake.drop(columns=["title", "subject","date"], axis=1) 
df_real=real.drop(columns=["title","subject","date"],axis=1)

print(df_fake.head(5))
print(df_real.head(5))

def mergeDatabase():
    # Sahte ve Gerçek verilerimizi tek bir veri setinde toplamamız gerekiyor. Bu nedenle ikisini birleştirip daha sonra karıştırma işlemi yapacağız.
        # birleştirme
    df_connect = pd.concat([df_fake, df_real], axis = 0) 
        # karıştırma
    df_connect = df_connect.sample(frac = 1) 
        # Indexleri sıfırlama (0'dan başlaması için)
    df_connect.reset_index(inplace = True) 
    df_connect.drop(["index"], axis = 1, inplace = True)

    # textProcessing methodunu kullanarak dataFrame'mizde bulunan cümleleri işliyoruz.
    df_connect["text"] = df_connect["text"].apply(sentenceProcessing)

    return df_connect