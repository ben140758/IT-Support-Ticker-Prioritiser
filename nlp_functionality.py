import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import string

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
Vector = list[float]
def preProcessText(raw_text: str) -> Vector:
    lowercase_text = raw_text.lower()
    lowercase_text.strip("\n")
    punctuation_removed_text = removePunctuation(lowercase_text)  
    tokenized_text = word_tokenize(punctuation_removed_text)   
    stopword_removed_text = removeStopWords(tokenized_text)    
    lemmatized_text = lemmatizeTokenizedText(stopword_removed_text)    
    return lemmatized_text

def removePunctuation(text: str) -> str:
    non_apostrophe_punctuation = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,"
    punctuation_removed_text = "".join([char for char in text if char not in non_apostrophe_punctuation])
    return punctuation_removed_text

def removeStopWords(text: list[str]) -> list[str]:
    stopword_removed_text = [word for word in text if word not in stopwords]
    return stopword_removed_text

def lemmatizeTokenizedText(text: list[str]) -> list[str]:
    #lemmatized_text = map(text, lemmatizer.lemmatize)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

def stemText(text):
    stemmed_text = [stemmer.stem(x) for x in text]
    return stemmed_text

def readTickets(ticketlist):
    return
def cleanTicket(ticket):
    #remove links
    return

def preprocess_text(text):
    new_text = text.lower()
    #new_text = re.sub('[^a-zA-Z]', ' ', new_text )
    new_text = re.sub(r'\s+', ' ', new_text)
    all_sentences = sent_tokenize(new_text)
    all_sentences = [re.sub('[^a-zA-Z-]', ' ',  x) for x in all_sentences]
    
    all_words = [word_tokenize(sent) for sent in all_sentences]
    all_words = [[y for y in x if len(y) > 2] for x in all_words]
    #bigrams = []
    """for sentence in all_words:
        for i, x in enumerate(sentence):
            try:
                bigram = x + ' ' + sentence[i + 1]
                bigrams.append(bigram)
            except IndexError:
                pass

    print(bigrams)"""
    
    
    all_words = [removeStopWords(x) for x in all_words]
    all_words = [lemmatizeTokenizedText(x) for x in all_words]
    all_words = [' '.join(x) for x in all_words]
    #all_words = [item for sublist in all_words for item in sublist]
    
    
    return all_words


def preprocess_text_bigrams(text):
    new_text = text.lower()
    #new_text = re.sub('[^a-zA-Z]', ' ', new_text )
    new_text = re.sub(r'\s+', ' ', new_text)
    all_sentences = sent_tokenize(new_text)
    all_sentences = [re.sub('[^a-zA-Z-]', ' ',  x) for x in all_sentences]
    
    all_words = [word_tokenize(sent) for sent in all_sentences]
    all_words = [[y for y in x if len(y) > 2] for x in all_words]
    
    bigrams = []
    for sentence in all_words:
        sentence_bigrams = []
        for i, x in enumerate(sentence):
            try:
                bigram = x + sentence[i + 1]
                sentence_bigrams.append(bigram)
            except IndexError:
                pass
        [bigrams.append(x) for x in sentence_bigrams]
        #bigrams.append(sentence_bigrams)
    #print(bigrams)
    """bigrams = [[y.split(' ') for y in x] for x in bigrams]
    bigrams = [[lemmatizeTokenizedText(word) for word in bigram] for bigram in bigrams]
    print(bigrams)"""
    
    
    #all_words = [[removeStopWords(y)] for x in all_words]
    #all_words = [lemmatizeTokenizedText(x) for x in all_words]
    
    
    return bigrams
    
if __name__ == "__main__":
    preprocess_text_bigrams("""I'm having trouble logging in to my E vision as it's saying my password is 
 incorrect so I'm therefore having trouble re-enrolling. 
 
 If there's anything you guys can do to help, I would greatly appreciate it. 
 
 
 All the best, """)
