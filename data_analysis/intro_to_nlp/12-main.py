#!/usr/bin/env python3

import pandas as pd
clean_text           = __import__('1-clean_text').clean_text
tokenize_text        = __import__('2-tokenize').tokenize_text
normalize_emoticons  = __import__('2-tokenize').normalize_emoticons
remove_stopwords     = __import__('3-remove_stopwords').remove_stopwords
filter_tokens        = __import__('4-filter_tokens').filter_tokens
normalize_tokens     = __import__('5-normalize_tokens').normalize_tokens
word2vec_embeddings  = __import__('12-word2vec').word2vec_embeddings

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned']         = df['message'].apply(clean_text)
df['tokens']          = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']          = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop']  = df['tokens'].apply(lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma']    = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()

X, model = word2vec_embeddings(corpus)

print(f"embedding matrix : {X.shape}  (one row per message)\n")

#  semantic relationships learned through the corpus 
for word in ['free', 'call', 'win', 'go']:
    if word in model.wv:
        similar = [w for w, _ in model.wv.most_similar(word, topn=5)]
        print(f"most similar to '{word}' : {similar}")

# OOV
oov_token = "xyzunseen"
in_vocab  = oov_token in model.wv
print(f"'{oov_token}' in vocab : {in_vocab}")
print(f"'free' in vocab      : {'free' in model.wv}\n")

# zero-vector messages
import numpy as np
zero_rows = (X == 0).all(axis=1).sum()
print(f"zero-vector messages : {zero_rows} / {len(X)}")
print(f"avg embedding norm   : {np.linalg.norm(X, axis=1).mean():.4f}")
