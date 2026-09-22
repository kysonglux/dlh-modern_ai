#!/usr/bin/env python3

import pandas as pd
from collections import Counter
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))

for idx in [2, 8, 100]:
    before = df.iloc[idx]['tokens_no_stop']
    after = filter_tokens(before)
    dropped = sorted(set(before) - set(after))
    print(f"[{df.iloc[idx]['label']}]")
    print(f"  before  : {before[:12]}")
    print(f"  after   : {after[:12]}")
    print(f"  dropped : {dropped}\n")


df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)

for step, col in [("tokenize", 'tokens'), ("stopwords", 'tokens_no_stop'),
                  ("filter", 'tokens_filtered')]:
    total = sum(len(t) for t in df[col])
    unique = len(set(t for tok in df[col] for t in tok))
    print(f"after {step:<12} : {total:>6} tokens  {unique:>5} unique")

filtered_set = set(t for tok in df['tokens_filtered'] for t in tok)
dropped_types = Counter(
    t for tok in df['tokens_no_stop'] for t in tok if t not in filtered_set
)
print("\ntop dropped:")
for w, c in dropped_types.most_common(10):
    print(f"  {w:<12} {c}")
