#!/usr/bin/env python3
""" a function pulls quotes from embedded JSON-LD on a page"""

import json
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """ a function pulls quotes from embedded JSON-LD on a page"""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    all = []

    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        try:
            data = json.loads(script.string)
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]

        for obj in items:
            if obj.get("@type") == "Quote":
                text = obj.get("text", "")
                author = obj.get("author", {}).get("name", "")

                raw = obj.get("keywords", [])
                if isinstance(raw, str):
                    tags = [t.strip() for t in raw.split(",") if t.strip()]
                else:
                    tags = raw

                all.append({
                    "text": text,
                    "author": author,
                    "tags": tags
                })
    return all
