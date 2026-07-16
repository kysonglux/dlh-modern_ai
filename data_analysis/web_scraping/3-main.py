#!/usr/bin/env python3
scrape_via_api = __import__('3-scrape_via_api').scrape_via_api
base = "https://quotes.toscrape.com"

quotes = scrape_via_api(base)
try:
    quotes = scrape_via_api(base)
    output = ""
    for i, q in enumerate(quotes):
        output += f"Quote #{i}:\n{q}\n"
    print(output)
except Exception as e:
    print(str(e))