#!/usr/bin/env python3
import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
plot_top_n_frequencies = __import__('7-freq').plot_top_n_frequencies

spam_keep_words = {"our", "from", "now", "your", "only"}

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])


# Full preprocessing pipeline
def preprocess(msg):
    t = tokenize_text(clean_text(msg), method="tweet")
    t = normalize_emoticons(t)
    t = remove_stopwords(t, keep_words=spam_keep_words)
    t = filter_tokens(t)
    t = normalize_tokens(t, method="stem")
    return t


df['tokens'] = df['message'].apply(preprocess)

# top-20 frequency distribution
fd_all = plot_top_n_frequencies(df['tokens'].tolist(), n=20)

# Per-class frequency distributions
spam_tokens = df[df['label'] == 'spam']['tokens'].tolist()
ham_tokens = df[df['label'] == 'ham']['tokens'].tolist()

print("\nTop 15 words in SPAM:")
fd_spam = plot_top_n_frequencies(spam_tokens, n=15)

print("\nTop 15 words in HAM:")
fd_ham = plot_top_n_frequencies(ham_tokens, n=15)

# Distinctive spam words
spam_total_tokens = sum(len(doc) for doc in spam_tokens)
ham_total_tokens = sum(len(doc) for doc in ham_tokens)

spam_unique = {}
for word, spam_count in fd_spam.most_common(50):
    ham_count = fd_ham.get(word, 0)
    spam_rate = spam_count / spam_total_tokens
    ham_rate = ham_count / ham_total_tokens if ham_count > 0 else 0.0001
    ratio = spam_rate / ham_rate
    if ratio > 5:
        spam_unique[word] = ratio

print("\nDISTINCTIVE SPAM WORDS:")
for word, ratio in sorted(spam_unique.items(),
                          key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {word:15}: {ratio:6.1f}x more frequent in spam")