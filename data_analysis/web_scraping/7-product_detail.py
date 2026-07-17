#!/usr/bin/env python3
"""open a detail page for on product, waits, and returns a dictionary"""

import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """open a detail page for on product, waits, and returns a dictionary"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    driver.get(url)
    time.sleep(delay)

    # --- Title: second <h4> inside .caption ---
    caption = driver.find_element("css selector", ".caption")
    h4s = caption.find_elements("tag name", "h4")
    title = h4s[1].text.strip() if len(h4s) > 1 else ""

    # --- Price: first <h4 class="price"> ---
    price = driver.find_element("css selector", "h4.price").text

    # --- Description: <p class="description"> ---
    description = driver.find_element("css selector", "p.description").text

    # --- Rating: count <p class="ws-icon ws-icon-star"> inside .ratings ---
    ratings = driver.find_element("css selector", ".ratings")
    stars = ratings.find_elements("css selector", ".ws-icon.ws-icon-star")
    rating = len(stars)

    driver.quit()

    return {
        "title": title,
        "price": price,
        "description": description,
        "rating": rating
    }
