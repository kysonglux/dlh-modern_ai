#!/usr/bin/env python3
""" scrapes the first page of quotes from website"""

from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """scrpes the first page of quotes from website"""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    results = []

    for each_block in soup.find_all("div", class_="quote"):
        text = each_block.find("span", class_="text").get_text(strip=True)
        author = each_block.find("small", class_="author").get_text(strip=True)

        tag_elements = each_block.find_all("a", class_="tag")
        tags = [tag.get_text(strip=True) for tag in tag_elements]

        results.append({
            "text": text,
            "author": author,
            "tags": tags
        })

    return results
