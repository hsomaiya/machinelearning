from datetime import datetime

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