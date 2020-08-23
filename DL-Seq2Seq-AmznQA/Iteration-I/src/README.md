Prerequisites:
- 
- All the programming libraries installed.
- Model checkpoints, tokenizer and embedding matrix available for the trained model.

Steps:
-
- Create a directory "checkpoints" and copy the model checkpoint in it.
- Copy tokenizer (qa_tokenizer) and embedding matrix (embedding_matrix.npy) in the root directory.
- Perform model inference using following code:

  ``` python main.py "Does this product come with any of the support tools or not?" ```
