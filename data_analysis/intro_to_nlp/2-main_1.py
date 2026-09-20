#!/usr/bin/env python3

import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)


ham_emo = df[(df['label'] == 'ham') &
    df['cleaned'].str.contains(r':\)|:\(|<3', regex=True)]['cleaned'].iloc[0]

tokens = tokenize_text(ham_emo)

print(f"raw tokens : {tokens[:10]}")
print(f"replace    : {normalize_emoticons(tokens[:10], emoticon_action='replace')}")
print(f"remove     : {normalize_emoticons(tokens[:10], emoticon_action='remove')}\n")


df['tokens'] = df['cleaned'].apply(
    lambda x: tokenize_text(x, method='tweet')
)

df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['token_count'] = df['tokens'].apply(len)

print(df.groupby('label')['token_count'].describe().round(1))
print(f"\nunique tokens : {len(set(t for tok in df['tokens'] for t in tok)):,}")