#!/usr/bin/env python3

import numpy as np
import pandas as pd
clean_text          = __import__('1-clean_text').clean_text
tokenize_text       = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords    = __import__('3-remove_stopwords').remove_stopwords
filter_tokens       = __import__('4-filter_tokens').filter_tokens
normalize_tokens    = __import__('5-normalize_tokens').normalize_tokens
bag_of_words        = __import__('10-bag_of_words').bag_of_words
tf_idf              = __import__('11-tf_idf').tf_idf

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned']         = df['message'].apply(clean_text)
df['tokens']          = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']          = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop']  = df['tokens'].apply(lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma']    = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()
labels = np.array(df['label'].tolist())
spam_mask = labels == 'spam'

X_bow,  v_bow  = bag_of_words(corpus)
X_tf,   v_tf   = tf_idf(corpus)

print(f"BoW shape    : {X_bow.shape}  dtype={X_bow.dtype}")
print(f"TF-IDF shape : {X_tf.shape}  dtype={X_tf.dtype}\n")

features_tf = np.array(v_tf.get_feature_names_out())
features_bw = np.array(v_bow.get_feature_names_out())

spam_tfidf_sum = np.asarray(X_tf[spam_mask].sum(axis=0)).flatten()
top_tfidf = features_tf[spam_tfidf_sum.argsort()[::-1][:15]]
print(f"top spam TF-IDF features : {list(top_tfidf)}\n")

idf = v_tf.idf_
low_idf_idx = idf.argsort()[:10]
print("lowest IDF (most common terms):")
for i in low_idf_idx:
    print(f"  {features_tf[i]:<25} idf={idf[i]:.3f}")