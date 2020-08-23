import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, max_ques_length, max_title_length, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding_ques = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                                        input_length=max_ques_length, 
                                                        weights=[embedding_matrix], 
                                                        trainable=False, mask_zero=True)
        self.embedding_title = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                                         input_length=max_title_length, 
                                                         weights=[embedding_matrix], 
                                                         trainable=False, mask_zero=True)
        
        self.lstm_ques = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
        self.lstm_title = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)
        
        self.dense_state_h_ques = tf.keras.layers.Dense(int(self.enc_units/2))
        self.dense_state_h_title = tf.keras.layers.Dense(int(self.enc_units/2))
        self.dense_state_c_ques = tf.keras.layers.Dense(int(self.enc_units/2))
        self.dense_state_c_title = tf.keras.layers.Dense(int(self.enc_units/2))

        self.conv2d_image = tf.keras.layers.Conv2D(32, (3, 3), 
                                    activation = 'relu',
                                    kernel_initializer = tf.keras.initializers.HeNormal(seed=42),
                                    padding = 'same')
        self.flatten_image = tf.keras.layers.Flatten()
        self.dense_image = tf.keras.layers.Dense(self.enc_units)

        self.dense_price = tf.keras.layers.Dense(self.enc_units)
        
    def call(self, question, title, price, image, hidden):
        # question shape == (batch_size, max_ques_length)
        # title shape == (batch_size, max_title_length)
        # price shape == (batch_size, 1)
        # image shape == (batch_size, img_width, img_height, channels)

        question_e = self.embedding_ques(question)
        # question_e shape == (batch_size, max_ques_length, embedding_dim)
        question_mask = self.embedding_ques.compute_mask(question)
        # question_mask shape == (batch_size, max_ques_length)
        output_ques, state_h_ques, state_c_ques = self.lstm_ques(question_e, initial_state = hidden, mask=question_mask)
        # output_ques shape == (batch_size, max_ques_length, enc_units)
        # state_h_ques shape == (batch_size, enc_units)
        # state_c_ques shape == (batch_size, enc_units)

        title_e = self.embedding_title(title)
        # title_e shape == (batch_size, max_title_length, embedding_dim)
        title_mask = self.embedding_title.compute_mask(title)
        # title_mask shape == (batch_size, max_title_length)
        output_title, state_h_title, state_c_title = self.lstm_title(title_e, initial_state = hidden, mask=title_mask)
        # output_title shape == (batch_size, max_title_length, enc_units)
        # state_h_title shape == (batch_size, enc_units)
        # state_c_title shape == (batch_size, enc_units)

        price = self.dense_price(price)
        # price shape == (batch_size, enc_units)

        image = self.conv2d_image(image)
        # image shape ==  (batch_size, img_width, img_height, filters=32)
        image = self.flatten_image(image)
        # image shape == (batch_size, 51200)
        image = self.dense_image(image)
        # image shape == (batch_size, enc_units)
        
        output = tf.concat([output_ques, output_title], axis=1)
        # output shape == (batch_size, max_ques_length + max_title_length, enc_units)
        output = tf.concat([tf.expand_dims(price, 1), output], axis=1)
        # output shape == (batch_size, max_ques_length + max_title_length + 1, enc_units)
        output = tf.concat([tf.expand_dims(image, 1), output], axis=1)
        # output shape == (batch_size, max_ques_length + max_title_length + 1 + 1, enc_units)

        state_h_ques = self.dense_state_h_ques(state_h_ques)
        # state_h_ques shape == (batch_size, int(enc_units/2))
        state_h_title = self.dense_state_h_title(state_h_title)
        # state_h_title shape == (batch_size, int(enc_units/2))
        state_h = tf.concat([state_h_ques, state_h_title], axis=-1)
        # state_h shape == (batch_size, enc_units)

        state_c_ques = self.dense_state_c_ques(state_c_ques)
        # state_c_ques shape == (batch_size, int(enc_units/2))
        state_c_title = self.dense_state_c_title(state_c_title)
        # state_c_title shape == (batch_size, int(enc_units/2))
        state_c = tf.concat([state_c_ques, state_c_title], axis=-1)
        # state_c shape == (batch_size, enc_units + enc_units)

        return output, state_h, state_c

    def initialize_hidden_state(self):
        return (tf.zeros((self.batch_sz, self.enc_units)),
                tf.zeros((self.batch_sz, self.enc_units)))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_ans_length, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                                   input_length=max_ans_length, 
                                                   weights=[embedding_matrix], 
                                                   trainable=False, mask_zero=True)
        
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = tf.keras.layers.AdditiveAttention()

        self.masking_hidden = tf.keras.layers.Masking()
        self.masking_enc_output = tf.keras.layers.Masking()
        
    def call(self, inputs, hidden, enc_output):
        # inputs shape == (batch_size, 1)
        # hidden shape == tuple of two (batch_size, enc_units)
        # enc_output shape == (batch_size, max_ques_length, enc_units)

        # hidden == lstm state_h
 
        hidden_with_time_axis = tf.expand_dims(hidden[0], 1)
        # hidden_with_time_axis shape == (batch_size, 1, enc_units)
        
        hidden_with_time_axis_mask = self.masking_hidden(hidden_with_time_axis)
        # hidden_with_time_axis_mask._keras_mask shape == (batch_size, 1)
        
        enc_output_mask = self.masking_enc_output(enc_output)
        # enc_output_mask._keras_mask shape == (batch_size, max_ques_length)
        
        context_vector = self.attention(inputs=[hidden_with_time_axis, enc_output],
                                        mask=[hidden_with_time_axis_mask._keras_mask, enc_output_mask._keras_mask])
        # context_vector shape == (batch_size, 1, enc_units)

        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(inputs)

        mask = self.embedding.compute_mask(inputs)
        # mask shape == (batch_size, 1)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([context_vector, x], axis=-1)
        
        # passing the concatenated vector to the LSTM
        output, state_h, state_c = self.lstm(x, initial_state=hidden, mask=mask)
        # output shape == (batch_size, 1, dec_units)
        # state_h shape == (batch_size, dec_units)
        # state_c shape == (batch_size, dec_units)
 
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size * 1, dec_units)
        
        x = self.fc(output)
        # output shape == (batch_size, vocab)
        
        return x, state_h, state_c

    def initialize_hidden_state(self):
        return (tf.zeros((self.batch_sz, self.dec_units)),
                tf.zeros((self.batch_sz, self.dec_units)))