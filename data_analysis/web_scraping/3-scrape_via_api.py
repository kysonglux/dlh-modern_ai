#!/usr/bin/env python3
""" fetches quote data from all the quotes' API pages """

import json
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """fetches quote data from all the quotes API pages """
    all = []
    page = 1

    while (1):
        api = f"{base_url}/api/quotes?page={page}"

        raw = fetch_html(api)

        data = json.loads(raw)

        for q in data.get("quotes", []):
            all.append({
                "text": q.get("text", ""),
                "author": q.get("author", {}).get("name", ""),
                "tags": q.get("tags", [])
            })

        if not data.get("has_next"):
            break
        page += 1
    return all
