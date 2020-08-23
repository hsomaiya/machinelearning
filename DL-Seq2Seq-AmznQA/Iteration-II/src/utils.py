from datetime import datetime
import urllib
import cv2
from numpy import array
import numpy as np

def timer(start_time=None):
  """ 
  Measure the block's execution time using the clock 
  """
  if not start_time:
    start_time = datetime.now()
    return start_time
  elif start_time:
    thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
    tmin, tsec = divmod(temp_sec, 60)
    print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
    
def get_lookup_maps(qa_tokenizer):
    qa_vocab = qa_tokenizer.word_index
    word2id = dict()
    id2word = dict()
    for k, v in qa_vocab.items():
        word2id[k] = v
        id2word[v] = k
    return word2id, id2word

def normalize(img):
  # Normalize pixel values to be between 0 and 1
  img = img / 255.0
  return img

def decode_image(url):
  url_response = urllib.request.urlopen(url)
  img_array = array(bytearray(url_response.read()), dtype=np.uint8)
  img = cv2.imdecode(img_array, -1)
  img = cv2.resize(img, (40, 40), interpolation=cv2.INTER_NEAREST)
  return img

def load_images(url):
  img = decode_image(url)
  img = normalize(img)
  return array(img)