#!/usr/bin/env python3
login_and_scrape = __import__('5-login_and_scrape').login_and_scrape
login_url = "https://quotes.toscrape.com/login"
try:
    quotes = login_and_scrape(login_url, user="foo", pwd="bar")
    output = ""
    for i, q in enumerate(quotes):
        output += f"Quote #{i}:\n{q}\n"
    print(output)
except Exception as e:
    print(str(e))