#!/usr/bin/env python3

import numpy as np
import pandas as pd
clean_text           = __import__('1-clean_text').clean_text
tokenize_text        = __import__('2-tokenize').tokenize_text
normalize_emoticons  = __import__('2-tokenize').normalize_emoticons
remove_stopwords     = __import__('3-remove_stopwords').remove_stopwords
filter_tokens        = __import__('4-filter_tokens').filter_tokens
normalize_tokens     = __import__('5-normalize_tokens').normalize_tokens
word2vec_embeddings  = __import__('12-word2vec').word2vec_embeddings
fasttext_embeddings  = __import__('13-fasttext').fasttext_embeddings

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned']         = df['message'].apply(clean_text)
df['tokens']          = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']          = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop']  = df['tokens'].apply(lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma']    = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()

X_w2v, m_w2v = word2vec_embeddings(corpus)
X_ft,  m_ft  = fasttext_embeddings(corpus)

print(f"Word2Vec  : {X_w2v.shape}")
print(f"FastText  : {X_ft.shape}\n")

# OOV handling
oov_tokens = ['freee', 'calll', 'prze', 'wnnr']
print(f"{'token':<12} {'in W2V vocab':>14} {'FastText has vector':>20}")
print("-" * 50)
for t in oov_tokens:
    in_w2v = t in m_w2v.wv
    print(f"  {t:<10} {str(in_w2v):>14} {'True':>20}")

# zero-vector messages
w2v_zeros = (X_w2v == 0).all(axis=1).sum()
ft_zeros  = (X_ft  == 0).all(axis=1).sum()
print(f"Word2Vec  zero-vector messages : {w2v_zeros}")
print(f"FastText  zero-vector messages : {ft_zeros}\n")

# similar words
for word in ['free', 'call', 'win']:
    if word in m_ft.wv:
        similar = [w for w, _ in m_ft.wv.most_similar(word, topn=5)]
        print(f"FastText most similar to '{word}' : {similar}")