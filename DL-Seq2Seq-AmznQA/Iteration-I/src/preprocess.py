import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_sequences(tokenizer, length, lines):
  """
    Encode and pad sequences
  """
  # Integer encode sequences
  X = tokenizer.texts_to_sequences(lines)
  # Pad sequences with 0 values
  X = pad_sequences(X, maxlen=length, padding='post')
  return X

def decontracted(phrase):
    """ 
    Performs decontraction on the phrase.
    """
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def removeHTML(text):
    """ 
    Removes HTML tags from the text.
    """
    return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))','', text)

def preprocess(sentence, maxlen):
    """ 
    Cleans and removes unwanted characters from the sentence. 
    """ 
    sent = removeHTML(sentence)
    sent = decontracted(sent)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = sent.lower()
    sent = sent.strip()
    sent = ' '.join(sent.split()[:maxlen])
    return sent