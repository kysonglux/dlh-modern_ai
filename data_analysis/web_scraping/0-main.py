#!/usr/bin/env python3
# Test 0: quotes.toscrape.com — basic fetch, expect title in HTML
fetch_html = __import__('0-fetch_html').fetch_html
url0 = "https://quotes.toscrape.com/"
html0 = fetch_html(url0, headers={"User-Agent": "test-agent"}, timeout=5)
print("Test 0:\n", html0[:209])

# Test 1: quote.toscrape.com — Non-existent URL
url1 = "https://quote.toscrape.com/"
try:
    html1 = fetch_html(url1)
except Exception as e:
    html1 = e

print("Test 1:\n", str(html1))


# Test 2: reddit.com — compare content length with and without User-Agent
url2 = "https://www.google.com/"
try:
    html2 = fetch_html(url2)
    html_default = len(html2)
except Exception as e:
    html_default = e

try:
    headers2 = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    html2 = fetch_html(url2, headers=headers2)
    html_browser = len(html2)
except Exception as e:
    html_browser = e

print(f"Test 2:\n {str(html_default)}\n {str(html_browser)}")

# Test 3: gamefaqs.com — blocked without UA, allowed with UA
url3 = "https://wikipedia.org"
try:
    html3 = fetch_html(url3)
    html_default = len(html3)
except Exception as e:
    html_default = e

try:
    headers3 = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    html3 = fetch_html(url3, headers=headers3)
    html_good = len(html3)
except Exception as e:
    html_good = e

print(f"Test 3:\n {str(html_default)}\n {str(html_good)}")
