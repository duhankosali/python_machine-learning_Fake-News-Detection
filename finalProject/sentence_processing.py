import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# İngilizce stopwordler listesini alır
stop_words = set(stopwords.words('english'))

def sentenceProcessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopwordleri kaldırır
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatized_text = ' '.join([lemmatizer.lemmatize(token) for token in tokens]) 
    
    # Remove unnecessary whitespace
    lemmatized_text = re.sub(' +', ' ', lemmatized_text)
    
    return lemmatized_text.strip()