import nltk
import pandas
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
import re

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()


def clean_text(text: str) -> str:
    # Strip HTML & XML
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    # Strip Hyperlinks & URLs
    text = re.sub(r'http\S+', r'<URL>', text)
    # Strip Non-word characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert string to lowercase
    text = text.replace('x', '')
    text = text.lower()
    return text

def tokenize_text(text):
    untokenized_sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [tokenize_sentence(sentence) for sentence in untokenized_sentences]
    tokens = [token for tokenized_sentence in tokenized_sentences for token in tokenized_sentence]
    return tokens


def tokenize_sentence(sentence):
    tokenized_sentence = nltk.word_tokenize(sentence)
    tokenized_sentence = [word for word in tokenized_sentence if len(word) > 2]
    return tokenized_sentence


def lemmatize_text(text: list):
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(lemmatized)

def stem_text(text: list):
    stemmed = [stemmer.stem(word) for word in text]
    return ' '.join(stemmed)




if __name__ == '__main__':
    print(stemmer.stem('connecting'))