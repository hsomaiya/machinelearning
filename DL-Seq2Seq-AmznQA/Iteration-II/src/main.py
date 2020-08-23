from inference import predict
import model
from model import Encoder, Decoder
from numpy import load, array
import os
import pickle
from preprocess import encode_sequences, preprocess, process_price
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf
from utils import get_lookup_maps, timer, load_images


if __name__ == "__main__":
    # Read question
    question = sys.argv[1]
    # Read title
    title = sys.argv[2]
    # Read price
    price = sys.argv[3]
    # Read image
    image = sys.argv[4]
    
    # Declaring constants
    max_ques_length = 11
    max_ans_length = 3 + 2 # +2 for start and end token
    max_title_length = 13
    embedding_dim = 300
    units = 1024
    BATCH_SIZE = 32

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
    
    # Preprocess title
    preprocessed_title = preprocess(title, max_title_length)

    # Encode and pad the text sequences
    title_encoded = encode_sequences(qa_tokenizer, max_title_length, [preprocessed_title])


    preprocessed_price = process_price(price)

    # Loading standardscalar
    with open(os.path.join(os.getcwd(),'sc'), 'rb') as handle:
        sc = pickle.load(handle)
    
    
    price_encoded = sc.transform(array(preprocessed_price).reshape(-1, 1))
    
    image_encoded = load_images(image)

    # Create lookup maps
    word2id, id2word = get_lookup_maps(qa_tokenizer)
    
    # Load word embeddings
    embedding_matrix = load(os.path.join(os.getcwd(),'embedding_matrix.npy'))




    # Create model objects
    encoder = Encoder(qa_vocab_size, embedding_dim, units, BATCH_SIZE, max_ques_length, max_title_length, embedding_matrix)
    decoder = Decoder(qa_vocab_size, embedding_dim, units, BATCH_SIZE, max_ans_length, embedding_matrix)

    # Define the optimizer and the loss function
    # optimizer = tf.keras.optimizers.Adam()

    # Checkpoints (Object-based saving)
    checkpoint_dir = os.path.join(os.getcwd(),'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder,
                                    decoder=decoder)

    # Restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Get predictions
    start_time = timer()
    predicted_ans = predict(encoder, decoder, question_encoded, title_encoded, price_encoded, image_encoded, max_ques_length, max_title_length, max_ans_length, word2id, id2word, units, beam_search=False)
    print("---- Without Beam Search ----")
    print("Original Question:", question)
    print("Predicted Answer:", predicted_ans)
    timer(start_time)

        
    start_time = timer()
    predicted_ans = predict(encoder, decoder, question_encoded, title_encoded, price_encoded, image_encoded, max_ques_length, max_title_length, max_ans_length, word2id, id2word, units, beam_search=True)
    print("---- With Beam Search ----")
    print("Original Question:", question)
    print("Predicted Answer:", predicted_ans)
    timer(start_time)