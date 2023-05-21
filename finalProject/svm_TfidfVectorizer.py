from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

from merge_database import mergeDatabase

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix

df_connect = mergeDatabase()
print(df_connect)

x = df_connect["text"]
y = df_connect["status"]


# Metni TF-IDF vektörlerine dönüştürür
vectorizer = TfidfVectorizer()
x_vectors = vectorizer.fit_transform(x)

# Veriyi eğitim, doğrulama ve test setlerine ayırır
x_train, x_test, y_train, y_test = train_test_split(x_vectors, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# SVM modelini oluşturur ve eğitir
model = svm.SVC()

# GridSearch için parametreleri belirler
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}

# GridSearchCV oluşturur ve eğitir
grid = GridSearchCV(model, param_grid, refit=True, verbose=2)
grid.fit(x_train, y_train)

# En iyi parametreler
print(grid.best_estimator_)

# Doğrulama setinde tahminler yapar
val_predictions = grid.predict(x_val)

# Tahminlerin ne kadar iyi olduğunu görmek için bir rapor yazdırır
print("Validation Set Metrics:")
print(classification_report(y_val, val_predictions))

# Test setinde tahminler yapar
test_predictions = grid.predict(x_test)

# Tahminlerin ne kadar iyi olduğunu görmek için bir rapor yazdırır
print("Test Set Metrics:")
print(classification_report(y_test, test_predictions))

# En iyi parametreler
print("Best Parameters: ", grid.best_params_)

# Accuracy scores
val_accuracy = accuracy_score(y_val, val_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Validation Accuracy: ", val_accuracy)
print("Test Accuracy: ", test_accuracy)

import pickle

# Modeli kaydedin
with open('SVM/svmModal.pkl', 'wb') as f:
     pickle.dump(grid, f)

# Confusion matrices
val_confusion_matrix = confusion_matrix(y_val, val_predictions)
test_confusion_matrix = confusion_matrix(y_test, test_predictions)

print("Validation Confusion Matrix: \n", val_confusion_matrix)
print("Test Confusion Matrix: \n", test_confusion_matrix)

# Plotting confusion matrix for the test set
disp = plot_confusion_matrix(grid, x_test, y_test, cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion Matrix for Test Set")
plt.savefig("SVM/graph/confusion_matrix_test.png")


