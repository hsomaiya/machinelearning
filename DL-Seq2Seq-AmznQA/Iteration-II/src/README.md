Prerequisites:
- 
- All the programming libraries installed.
- Model checkpoints, tokenizer, embedding matrix, standard scalar available for the trained model.

Steps:
-
- Create a directory "checkpoints" and copy the "attention model with a mask generation layer" checkpoint in it.
- Copy tokenizer (qa_tokenizer), embedding matrix (embedding_matrix.npy) and standard scalar (sc) in the root directory.
- Perform model inference using following code:

  ``` python main.py 'Are these hex head screws?' 'Dynamite screw set slash stampede rustler bandit' '$25.99' 'https://images-na.ssl-images-amazon.com/images/I/41qZd9n61AL._SX38_SY50_CR,0,0,38,50_.jpg' ```
