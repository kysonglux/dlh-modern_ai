#!/usr/bin/env python3

import pandas as pd
from collections import Counter
clean_text          = __import__('1-clean_text').clean_text
tokenize_text       = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords    = __import__('3-remove_stopwords').remove_stopwords

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens']  = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens']  = df['tokens'].apply(normalize_emoticons)
spam_keep_words = {"won", "our", "from", "now", "your", "only"}

# NLTK blindly removes spam-signal words but sometimes some of these words
# could be spam indicative so we need to first look for commun words for both spam and ham messages 
spam_kw = df[
    (df['label'] == 'spam') &
    df['tokens'].apply(lambda t: any(w in t for w in spam_keep_words))
]['tokens'].iloc[3]

print(f"spam tokens         : {spam_kw[:15]}")
print(f"without keep_words  : {remove_stopwords(spam_kw[:15])}")
print(f"with SPAM_KEEP_WORDS: {remove_stopwords(spam_kw[:15], keep_words=spam_keep_words)}")
rescued = sorted(set(remove_stopwords(spam_kw[:15], keep_words=spam_keep_words)) -
                 set(remove_stopwords(spam_kw[:15])))
print(f"rescued             : {rescued}\n")

df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words)
)

tok_before = sum(len(t) for t in df['tokens'])
tok_after  = sum(len(t) for t in df['tokens_no_stop'])
voc_before = set(t for tok in df['tokens']         for t in tok)
voc_after  = set(t for tok in df['tokens_no_stop'] for t in tok)
print(f"tokens : {tok_before:,} -> {tok_after:,}  ({(1-tok_after/tok_before)*100:.1f}% reduction)")
print(f"vocab  : {len(voc_before):,} -> {len(voc_after):,}  ({(1-len(voc_after)/len(voc_before))*100:.1f}% reduction)\n")

removed = Counter(t for tok in df['tokens'] for t in tok) \
        - Counter(t for tok in df['tokens_no_stop'] for t in tok)
print("top removed:")
for w, c in removed.most_common(10):
    print(f"  {w:<12} {c}")
    