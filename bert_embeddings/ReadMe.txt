This is a place to cache embeddings performed on the entire corpus

Running BERT (or Sentence Bert) embeddings on the whole corpus takes a long time.
It is better to do this once, save them in this directory, and re-use them every time.

Since the embeddings never change, there should never be a need to re-fresh these again.

If you DO need to refresh them, run refresh_model_embeddings() in word_embeddings.py

You *should* never need to do this.  WHen the models are created and trained, they automatically
will refresh the embeddings if they are not present