import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional

from sklearn.model_selection import train_test_split
from merge_database import mergeDatabase # Fake.csv ve Real.csv veritabanlarımı birleştiriyor.

# Modeli geliştirmek için ekledik.
from tensorflow.keras.callbacks import EarlyStopping



df_connect = mergeDatabase()

print(df_connect)

x = df_connect["text"]
y = df_connect["status"]

# Veri setimizi eğitim, doğrulama ve test veri setlerine ayırıyoruz
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5)

# Tokenizer ve padding işlemleri
max_vocab = 20000
max_len = 250
tokenizer = Tokenizer(num_words=max_vocab, lower=True)
tokenizer.fit_on_texts(x_train)

# Tokenizer Kaydet
import pickle

# with open('DeepLearning/Tokenizer/tokenizer.pkl', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_val_seq = tokenizer.texts_to_sequences(x_val)

x_train_pad = pad_sequences(x_train_seq, maxlen=max_len, padding='post')
x_test_pad = pad_sequences(x_test_seq, maxlen=max_len, padding='post')
x_val_pad = pad_sequences(x_val_seq, maxlen=max_len, padding='post')

# LSTM modeli
model = Sequential()
model.add(Embedding(max_vocab, 128, input_length=max_len))
model.add(Bidirectional(LSTM(32, return_sequences=True)))  # return_sequences=True ekleyerek daha fazla LSTM katmanı eklememize izin verir
model.add(Dropout(0.5))  # Dropout miktarını artırdık
model.add(LSTM(32))  # İkinci bir LSTM katmanı ekledik
model.add(Dropout(0.5))  # İkinci Dropout katmanı
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping callback'ini kullanın
early_stop = EarlyStopping(monitor='val_loss', patience=3)  # 3 epoch boyunca iyileşme olmazsa eğitimi durdurur

# Model eğitimi
epochs = 7
batch_size = 32
history = model.fit(x_train_pad, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(x_val_pad, y_val), 
                    callbacks=[early_stop])  # callbacks parametresine early_stop ekledik

# Model değerlendirmesi
test_loss, test_acc = model.evaluate(x_test_pad, y_test)
print("Test doğruluk oranı şu şekildedir: ", test_acc)

#Modeli kaydetme
model.save('DeepLearning/lstm_model_best.h5')


# GRAFİKLEŞTİRME

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import os

# Save directory
save_dir = 'DeepLearning/graph'
os.makedirs(save_dir, exist_ok=True)

# Eğitim ve Doğrulama Kaybı (Loss) Grafiği
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss_graph_best.png'))  # Save the figure
plt.close()  # Close the figure to free up memory
print('Eğitim ve doğrulama (LOSS) grafiği oluşturuldu.')

# Eğitim ve Doğrulama Doğruluk (Accuracy) Grafiği
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_dir, 'accuracy_graph_best.png'))  # Save the figure
plt.close()  # Close the figure to free up memory
print('Eğitim ve doğrulama (Accuracy) grafiği oluşturuldu.')

# Karışıklık Matrisi (Confusion Matrix)
y_pred = model.predict(x_test_pad).round()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(os.path.join(save_dir, 'confusion_matrix_best.png'))  # Save the figure
plt.close()  # Close the figure to free up memory
print('Karışıklık matrisi grafiği oluşturuldu.')
