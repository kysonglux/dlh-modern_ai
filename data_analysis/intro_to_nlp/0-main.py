#!/usr/bin/env python3
import csv
import pandas as pd
explore_data = __import__('0-explore_data').explore_data


df = pd.read_csv('SMSSpamCollection', sep='\t',
                 names=['label', 'message'])
# Basic exploration of the cleaned dataset
print("Dataset Overview:")
print(f"Total messages: {len(df)}\n")

print("Class distribution:")
print((df['label'].value_counts(normalize=True) * 100).round(2))

df['msg_length'] = df['message'].str.len()
print("\nMessage length statistics:")
print(df.groupby('label')['msg_length'].describe())

explore_data(df)

placeholder_mask = df['message'].str.contains(
    r'<#>|<decimal>|<time>|<url>|<email>', regex=True)
print(f"\nDataset placeholders (<#>, <decimal> …): {placeholder_mask.sum()}")

phone_mask = df['message'].str.contains(r'\+?\d[\d\s\-]{6,}\d', regex=True)
print(f"Messages containing phone-like numbers : {phone_mask.sum()}")

url_mask = df['message'].str.contains(r'https?://\S+|www\.\S+', regex=True)
print(f"Messages containing URLs               : {url_mask.sum()}")

currency_mask = df['message'].str.contains(r'[£$€]\d+', regex=True)
print(f"Messages containing currency amounts   : {currency_mask.sum()}")

unicode_mask = df['message'].str.contains(
    r'[\u2018\u2019\u201c\u201d\u2014\u2013\u2026]', regex=True)
print(f"Messages with Unicode punctuation      : {unicode_mask.sum()}")

emoticon_mask = df['message'].str.contains(
    r':\)|:\(|:-\)|:-\(|:D|;\)|<3', regex=True)
print(f"Messages containing ASCII emoticons    : {emoticon_mask.sum()}")

rep_mask = df['message'].str.contains(r'[!?]{2,}', regex=True)
print(f"Messages with repeated punctuations    : {rep_mask.sum()}")
