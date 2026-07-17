#!/usr/bin/env python3
""" a function logs in and scrapes quotes visible after authentication"""

import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """a function logs in and scrapes quotes visible after authentication """
    session = requests.Session()

    html = session.get(login_url)
    soup = BeautifulSoup(html.text, "html.parser")

    try:
        csrf_token = soup.find("input", {"name": "csrf_token"})["value"]
    except Exception as e:
        print(e)

    payload = {
        "username": user,
        "password": pwd,
        "csrf_token": csrf_token
    }

    session.post(login_url, data=payload)

    protected_page = session.get("https://quotes.toscrape.com/")
    soup = BeautifulSoup(protected_page.text, "html.parser")

    results = []

    for block in soup.find_all("div", class_="quote"):
        text = block.find("span", class_="text").get_text(strip=True)
        author = block.find("small", class_="author").get_text(strip=True)

        tag_elements = block.find_all("a", class_="tag")
        tags = [t.get_text(strip=True) for t in tag_elements]

        results.append({
            "text": text,
            "author": author,
            "tags": tags
        })

    return results
