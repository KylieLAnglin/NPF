import string
import unicodedata

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


def remove_punctuation(input_string):
    """
    This function removes all punctuation from a given string
    """
    # Create a string of all punctuation characters
    punctuations = string.punctuation
    
    # Remove all punctuation from the input string using list comprehension
    no_punctuations = ''.join([char for char in input_string if char not in punctuations])
    
    return no_punctuations


def remove_non_unicode(input_string):
    """
    This function removes all non-unicode characters from a given string
    """
    # Remove non-unicode characters using unicodedata.normalize
    normalized_string = unicodedata.normalize('NFKD', input_string)
    non_unicode = normalized_string.encode('ASCII', 'ignore').decode('ASCII')
    
    return non_unicode


def lemmatize_string(input_string):
    """
    This function lemmatizes all words in a given string
    """
    # Tokenize the input string into individual words
    words = word_tokenize(input_string)
    
    # Initialize the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each word in the input string using a list comprehension
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the lemmatized words back into a single string
    lemmatized_string = ' '.join(lemmatized_words)
    
    return lemmatized_string

def stem_string(input_string):
    """
    This function stems all words in a given string using the SnowballStemmer.
    """
    # Tokenize the input string into individual words
    words = word_tokenize(input_string)
    
    # Initialize the SnowballStemmer with the 'english' language
    stemmer = SnowballStemmer('english')
    
    # Stem each word in the input string using a list comprehension
    stemmed_words = [stemmer.stem(word) for word in words]
    
    # Join the stemmed words back into a single string
    stemmed_string = ' '.join(stemmed_words)
    
    return stemmed_string

