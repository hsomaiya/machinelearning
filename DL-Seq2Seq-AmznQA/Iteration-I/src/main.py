from inference import predict
import model
from model import Encoder, Decoder
from numpy import argmax, array, asarray, zeros, save, load
import os
import pickle
from preprocess import encode_sequences, preprocess
import sys
import tensorflow as tf
from utils import get_lookup_maps, timer


if __name__ == "__main__":
    # Read question
    question = sys.argv[1]
    
    # Declaring constants
    max_ques_length = 11
    max_ans_length = 3 + 2 # +2 for start and end token
    embedding_dim = 300
    units = 1024
    BATCH_SIZE = 512

    # Suppress verbosity
    tf.get_logger().setLevel('ERROR')
	
    # Preprocess question
    preprocessed_question = preprocess(question, max_ques_length)
    
    # Loading tokenizer
    with open(os.path.join(os.getcwd(),'qa_tokenizer'), 'rb') as handle:
        qa_tokenizer = pickle.load(handle)
    
    # Find vocab size
    qa_vocab_size = len(qa_tokenizer.word_index) + 1

    # Encode and pad the text sequences
    question_encoded = encode_sequences(qa_tokenizer, max_ques_length, [preprocessed_question])
    
    # Create lookup maps
    word2id, id2word = get_lookup_maps(qa_tokenizer)
    
    # Load word embeddings
    embedding_matrix = load(os.path.join(os.getcwd(),'embedding_matrix.npy'))

    # Create model objects
    encoder = Encoder(qa_vocab_size, embedding_dim, units, BATCH_SIZE, max_ques_length, embedding_matrix)
    decoder = Decoder(qa_vocab_size, embedding_dim, units, BATCH_SIZE, max_ques_length, embedding_matrix)

    # Checkpoints (Object-based saving)
    checkpoint_dir = os.path.join(os.getcwd(),'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                    decoder=decoder)

    # Restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Get predictions
    start_time = timer()
    predicted_ans = predict(encoder, decoder, question_encoded, max_ques_length, max_ans_length, word2id, id2word, units, beam_search=False)
    print("---- Without Beam Search ----")
    print("Original Question:", question)
    print("Predicted Answer:", predicted_ans)
    timer(start_time)

        
    start_time = timer()
    predicted_ans = predict(encoder, decoder, question_encoded, max_ques_length, max_ans_length, word2id, id2word, units, beam_search=True)
    print("---- With Beam Search ----")
    print("Original Question:", question)
    print("Predicted Answer:", predicted_ans)
    timer(start_time)