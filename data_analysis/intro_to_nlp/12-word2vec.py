#!/usr/bin/env python3
"""train Word2Vec and return per-message embeddings."""
import numpy as np
import gensim.models


def word2vec_embeddings(corpus_tokens, vector_size=100,
                        window=5, min_count=2, sg=0, epochs=10, workers=4):
    """Train Word2Vec and return per-message embeddings."""
    if not isinstance(corpus_tokens, list) or not corpus_tokens:
        return
    if not isinstance(vector_size, int) or vector_size < 1:
        raise ValueError("vector_size must be a positive integer.")
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    if not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer.")
    if not isinstance(sg, int) or sg not in [0, 1]:
        raise ValueError("sg must be either 0 (CBOW) or 1 (Skip-gram).")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")

    model = gensim.models.Word2Vec(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
    )

    embeddings = []
    for tokens in corpus_tokens:
        token_vectors = [model.wv[token]
                         for token in tokens if token in model.wv]
        if token_vectors:
            message_embedding = np.mean(token_vectors,
                                        axis=0).astype(np.float64)
        else:
            message_embedding = np.zeros(vector_size)
        embeddings.append(message_embedding)
    x = np.array(embeddings)

    return x, model
