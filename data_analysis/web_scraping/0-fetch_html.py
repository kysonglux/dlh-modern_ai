#!/usr/bin/env python3
"""a function that fetches a web page and returns its HTML as text"""

import requests


def fetch_html(url, headers=None, timeout=10):
    """ a function that fetches a webpage and returns its HTML"""
    response = requests.get(url, headers=headers, timeout=timeout)

    if response.status_code >= 400:
        response.raise_for_status()

    return response.text
