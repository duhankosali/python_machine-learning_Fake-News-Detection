# Kullanıcı

from sklearn.feature_extraction.text import TfidfVectorizer # Vektörize etme kütüphanesi
import pickle # Önceden kaydetmiş olduğum scriptleri kullanabilmek için.
from sentence_processing import sentenceProcessing
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# rfcModel_best.pkl dosyasını açın --> test, train ve doğrulama var ayrıca bazı hiperparametreler test ediliyor.
with open('RandomForest/rfcModal_best.pkl', 'rb') as f:
    rfcModal = pickle.load(f)

# SVM Modeli
# with open('SVM/svmModal.pkl', 'rb') as f:
#     svmModal = pickle.load(f)

# LSTM modeli
lstmModal = load_model('DeepLearning/lstm_model_best.h5') 
max_len = 250 # Bu değer, modelinizi eğitirken kullandığınız max_len değeri ile aynı olmalıdır.

# Modellerimizi çağırdık. Şimdi de vectorize yöntemlerimizi çağırmamız gerekiyor.

# Random Forest Vectorizer

with open('Vectorizer/CountVectorizer.pkl', 'rb') as f: # Count Vectorizer kullan
    countVectorize = pickle.load(f)

with open('Vectorizer/TfidfVectorizer.pkl', 'rb') as f: # TF-IDF Vectorizer kullan
    tfIdfVectorize = pickle.load(f)

# LSTM Vectorizer

with open('DeepLearning/Tokenizer/tokenizer.pkl', 'rb') as f: # Tokenizer Kullan 
    tokenizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sonuc', methods=['POST'])
def sonuc():
    user_text = request.form['news']
    chosen_algo = request.form['algo']
    chosen_vec = request.form['vec']

    processed_text = sentenceProcessing(user_text) # Kullanıcıdan aldığımız girdiyi ayıklıyoruz. (Metin işleme -- NLP)

    if chosen_algo == "LSTM":
        # Metni vektörel biçime getir
        tokenized_text = tokenizer.texts_to_sequences([processed_text])
        padded_text = pad_sequences(tokenized_text, maxlen=max_len, padding='post')
        # Tahmin
        prediction = lstmModal.predict(padded_text)
        print("LSTM")
    else:
        # Vektörize et (TF IDF kullanarak)
        if chosen_vec == "count":
            vectorized_text = countVectorize.transform([processed_text]) # NLP ile işlediğimiz girdiyi vektörel biçime getiriyoruz.
            print("Count Vectorizer")
        else:
            vectorized_text = tfIdfVectorize.transform([processed_text]) # NLP ile işlediğimiz girdiyi vektörel biçime getiriyoruz.
            print("TF-IDF Vectorizer")
        # Tahmin
        prediction = rfcModal.predict(vectorized_text) # Vektörel biçimdeki girdimizi modelimize gönderiyoruz ve burada tahmin işlemi gerçekleşiyor. (0 veya 1 olarak bir dönüt alacağız.)
        print("RFC")

    print("Girilen Metin: ", user_text)
    print("Düzenlenmiş: ", processed_text)

    if prediction == 0:
        analyze_text = "Girilen metin sahte bir haber olabilir."

    else:
        analyze_text = "Girilen metin gerçek bir haber olabilir."

    return render_template('sonuc.html', input_text=user_text, input_algo=chosen_algo, analyze_text=analyze_text)

if __name__ == '__main__':
    app.run(debug=True)
############## Web Proje Sonu