#!/usr/bin/env python3

import pandas as pd
import nltk
from collections import Counter
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)

test_tokens = ['running', 'runs', 'ran', 'runner', 'winning', 'won', 'called', 'calling']
print(f"input     : {test_tokens}")
print(f"lemmatize : {normalize_tokens(test_tokens, method='lemmatize')}")
print(f"stem      : {normalize_tokens(test_tokens, method='stem')}\n")

lemmatizer = nltk.stem.WordNetLemmatizer()
for tokens in [["won", "winning"], ["running"], ["better"]]:
    naive = [lemmatizer.lemmatize(t) for t in tokens]
    pos_aw = normalize_tokens(tokens, method="lemmatize")
    for t, n, p in zip(tokens, naive, pos_aw):
        print(f"{t:<10} naive={n:<10} pos-aware={p}")

df['tokens_lemma'] = df['tokens_filtered'].apply(normalize_tokens)
df['tokens_stem']  = df['tokens_filtered'].apply(lambda t: normalize_tokens(t, method='stem'))

for label, col in [("filtered", 'tokens_filtered'), ("lemmatize", 'tokens_lemma'), ("stem", 'tokens_stem')]:
    print(f"{label:<12} : {len(set(t for tok in df[col] for t in tok)):>5} unique tokens")

for cls in ['spam', 'ham']:
    counts = Counter(t for tok in df[df.label == cls]['tokens_lemma'] for t in tok)
    print(f"top [{cls}] : {[w for w, _ in counts.most_common(10)]}")