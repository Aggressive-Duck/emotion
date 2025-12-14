import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Initialize resources
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
word_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

def remove_punctuation(text):
    if not isinstance(text, str):
        return str(text)
    return text.translate(str.maketrans('', '', string.punctuation))

def replace_space(text):
    if not isinstance(text, str):
        return str(text)
    return text.replace("\n", " ")

def remove_stopword(text):
    if not isinstance(text, str):
        return str(text)
    return ' '.join([word for word in text.split() if word not in STOPWORDS])

def remove_specific_words(text, words_to_remove):
    """
    Removes a specific set of words (like frequent or rare words) from the text.
    """
    if not isinstance(text, str):
        return str(text)
    if not words_to_remove:
        return text
    return ' '.join([word for word in text.split() if word not in words_to_remove])

def lemmatize_words(text):
    if not isinstance(text, str):
        return str(text)
    try:
        # NLTK pos_tag expects a list of words
        words = text.split()
        if not words:
            return ""
        pos_text = pos_tag(words)
        return ' '.join([lemmatizer.lemmatize(word, word_map.get(pos[0][0], wordnet.NOUN)) for word, pos in pos_text])
    except Exception as e:
        print(f"Error in lemmatization: {e}")
        return text

def full_preprocess_pipeline(text, frequent_words=None, rare_words=None):
    """
    Runs the full pipeline on a single string of text.
    """
    text = text.lower()
    text = remove_punctuation(text)
    text = replace_space(text)
    text = remove_stopword(text)
    
    if frequent_words:
        text = remove_specific_words(text, frequent_words)
    
    if rare_words:
        text = remove_specific_words(text, rare_words)
        
    text = lemmatize_words(text)
    return text
