#!/usr/bin/env python3

import pandas as pd
clean_text = __import__('1-clean_text').clean_text

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)

for msg in df[df['message'].str.contains('<#>', regex=False)]['message'].iloc[1:3]:
    print(f"before : {msg}")
    print(f"after  : {clean_text(msg)}\n")

# URL & number replacement
for msg in df[(df['label'] == 'spam') & df['message'].str.contains('http|www', regex=True)]['message'].iloc[:1]:
    print(f"before : {msg[:100]}")
    print(f"after  : {clean_text(msg)[:100]}\n")

for msg in df[(df['label'] == 'spam') & df['message'].str.contains(r'\d{8,}', regex=True)]['message'].iloc[:2]:
    print(f"before : {msg[:100]}")
    print(f"after  : {clean_text(msg)[:100]}\n")

print("Phone patterns matched first to avoid partial replacement by the general digit pass.\n")

# Unicode punctuation — curly quotes and ellipses still present after preclean
for msg in df[df['message'].str.contains('[''…]', regex=True)]['message'].iloc[:2]:
    print(f"before : {msg}")
    print(f"after  : {clean_text(msg)}\n")


# emoji (dataset has only emoticons)
msg = "Hey! 😊 Great offer 🎉 Call now 📞"
for action in ["replace", "remove", "keep"]:
    print(f"[{action:7}] {clean_text(msg, emoji_action=action)}")


df['orig_len']    = df['message'].str.len()
df['cleaned_len'] = df['cleaned'].str.len()
print(df.groupby('label')[['orig_len', 'cleaned_len']].mean().round(1))