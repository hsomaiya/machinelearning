from numpy import argsort, expand_dims
import tensorflow as tf

def evaluate_beam(inputs_ques, inputs_title, inputs_price, inputs_image, encoder, decoder, max_length_inp_ques, max_length_inp_title, max_length_targ, word2id, id2word, units, beam_index=3):
    """
      inputs_ques type = numpy.ndarray 
      inputs_ques shape = (1, max_length_inp_ques)
      inputs_title type = numpy.ndarray 
      inputs_title shape = (1, max_length_inp_title)
      inputs_price type = numpy.ndarray 
      inputs_price shape = (1, 1)
      inputs_image type = numpy.ndarray 
      inputs_image shape = (1, image_width, image_height, channels)
    """

    sentence = ''
    for i in inputs_ques[0]:
        if i == 0: # break if post padding detected
            break
        sentence = sentence + id2word[i] + ' '
    inputs_ques = tf.convert_to_tensor(inputs_ques)
    inputs_title = tf.convert_to_tensor(inputs_title)
    inputs_price = tf.convert_to_tensor(inputs_price)
    inputs_image = tf.convert_to_tensor(inputs_image)

    start = [word2id['<start>']]
    
    # result[0][0] = index of the starting word
    # result[0][1] = probability of the word predicted
    result = [[start, 0.0]]

    enc_hidden = (tf.zeros((1, units)), tf.zeros((1, units)))
    enc_outputs = encoder(inputs_ques, inputs_title, inputs_price, inputs_image, enc_hidden)
    enc_output = enc_outputs[0]
    enc_states = enc_outputs[1:]
    dec_state_h, dec_state_c = enc_states
    dec_input = tf.expand_dims([word2id['<start>']], 0)

    # (max_length_targ-1) because start token is already added in the result
    while len(result[0][0]) < (max_length_targ-1):
        temp = []
        for s in result:

          predictions, dec_state_h, dec_state_c = \
                      decoder(dec_input, (dec_state_h, dec_state_c), enc_output)

         
          
          # Getting the top <beam_index>(n) predictions
          word_preds = argsort(predictions[0])[-beam_index:]
          
          # creating a new list so as to put them via the model again
          for w in word_preds:       
            next_cap, prob = s[0][:], s[1]
            next_cap.append(w)
            prob += predictions[0][w]
            temp.append([next_cap, prob])
        result = temp
        # Sorting according to the probabilities
        result = sorted(result, reverse=False, key=lambda l: l[1])
        # Getting the top words
        result = result[-beam_index:]
        
        predicted_id = result[-1] # with Max Probability
        pred_list = predicted_id[0]
        
        prd_id = pred_list[-1] 

        if(prd_id!=word2id['<end>']):
          dec_input = tf.expand_dims([prd_id], 0)  # Decoder input is the word predicted with highest probability among the top_k words predicted
        else:
          break

    result = result[-1][0]
    
    intermediate_result = [id2word[i] for i in result]
    
    final_result = []
    for i in intermediate_result:
        if i != '<end>':
            final_result.append(i)
        else:
            break

    
    final_result = ' '.join(final_result[1:])
    return final_result, sentence

def evaluate(inputs_ques, inputs_title, inputs_price, inputs_image, encoder, decoder, max_length_inp_ques, max_length_inp_title, max_length_targ, word2id, id2word, units):
    """
      inputs_ques type = numpy.ndarray 
      inputs_ques shape = (1, max_length_inp_ques)
      inputs_title type = numpy.ndarray 
      inputs_title shape = (1, max_length_inp_title)
      inputs_price type = numpy.ndarray 
      inputs_price shape = (1, 1)
      inputs_image type = numpy.ndarray 
      inputs_image shape = (1, image_width, image_height, channels)
    """
    
    sentence = ''
    for i in inputs_ques[0]:
        if i == 0: # break if post padding detected
            break
        sentence = sentence + id2word[i] + ' '
    inputs_ques = tf.convert_to_tensor(inputs_ques)
    inputs_title = tf.convert_to_tensor(inputs_title)
    inputs_price = tf.convert_to_tensor(inputs_price)
    inputs_image = tf.convert_to_tensor(inputs_image)
    
    result = ''
    enc_hidden = (tf.zeros((1, units)), tf.zeros((1, units)))
    enc_outputs = encoder(inputs_ques, inputs_title, inputs_price, inputs_image, enc_hidden)
    enc_output = enc_outputs[0]
    enc_states = enc_outputs[1:]
    dec_state_h, dec_state_c = enc_states

    dec_input = tf.expand_dims([word2id['<start>']], 0)
    for t in range(max_length_targ): # limit the length of the decoded sequence
        predictions, dec_state_h, dec_state_c = \
                      decoder(dec_input, (dec_state_h, dec_state_c), enc_output)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += id2word[predicted_id] + ' '
        if id2word[predicted_id] == '<end>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence

def process(str):
  str = str.replace('<start>','')
  str = str.replace('<end>','')
  return str.strip()

def predict(encoder, decoder, random_input_ques, random_input_title, random_input_price, random_input_image, max_ques_length, max_title_length, max_ans_length, word2id, id2word, units, beam_search=False):   
    random_input_image = expand_dims(random_input_image, 0)
    if beam_search:
        predicted_ans, ques = evaluate_beam(random_input_ques, random_input_title, random_input_price, random_input_image, encoder, decoder, max_ques_length, max_title_length, max_ans_length, word2id, id2word, units)
    else:
        predicted_ans, ques = evaluate(random_input_ques, random_input_title, random_input_price, random_input_image, encoder, decoder, max_ques_length, max_title_length, max_ans_length, word2id, id2word, units)
    return process(predicted_ans)
