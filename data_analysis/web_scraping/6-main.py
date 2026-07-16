#!/usr/bin/env python3
scrape_products = __import__('6-products_list').scrape_products_list
url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"
products = scrape_products(url)
try:
    products = scrape_products(url)
    output = ""
    for i, q in enumerate(products):
        output += f"Product #{i}:\n{q}\n"
    print(output)
except Exception as e:
    print(str(e))