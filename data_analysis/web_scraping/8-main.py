#!/usr/bin/env python3
scroll_and_scrape = __import__(
    '8-scroll_and_scrape').scroll_and_scrape
url = "https://webscraper.io/test-sites/e-commerce/scroll/computers/laptops"
try:
    products = scroll_and_scrape(url)
    output = ""
    for i, q in enumerate(products):
        output += f"Product #{i}:\n{q}\n"
    print(output)
except Exception as e:
    print(str(e))
