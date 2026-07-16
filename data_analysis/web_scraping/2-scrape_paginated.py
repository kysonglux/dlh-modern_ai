#!/usr/bin/env python3
""" a funtion scrape until no more pages remain"""

from bs4 import BeautifulSoup
import time
from urllib import parse
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """ a function scape until no more pages remain"""
    all = []

    while (1):
        html = fetch_html(base_url)
        soup = BeautifulSoup(html, "html.parser")

        page = scrape_basic(base_url)
        all.extend(page)

        next_li = soup.find("li", class_="next")
        if not next_li:
            break
        next_href = next_li.find("a")["href"]

        base_url = parse.urljoin(base_url, next_href)

        time.sleep(1)

    return all
