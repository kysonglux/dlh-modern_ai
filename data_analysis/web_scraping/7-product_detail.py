#!/usr/bin/env python3
"""open a detail page for on product, waits delay seconds, and returns a dictionary"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def scrape_product_detail(url, delay=2.0):
    """open a detail page for on product, waits delay seconds, and returns a dictionary"""
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
    caption = driver.find_element(By.CSS_SELECTOR, ".caption")
    h4s = caption.find_elements(By.TAG_NAME, "h4")
    title = h4s[1].text.strip() if len(h4s) > 1 else ""

    # --- Price: first <h4 class="price"> ---
    price = driver.find_element(By.CSS_SELECTOR, "h4.price").text.strip()

    # --- Description: <p class="description"> ---
    description = driver.find_element(By.CSS_SELECTOR, "p.description").text.strip()

    # --- Rating: count <p class="ws-icon ws-icon-star"> inside .ratings ---
    rating_block = driver.find_element(By.CSS_SELECTOR, ".ratings")
    stars = rating_block.find_elements(By.CSS_SELECTOR, "p.ws-icon.ws-icon-star")
    rating = len(stars)

    driver.quit()

    return {
        "title": title,
        "price": price,
        "description": description,
        "rating": rating
    }
