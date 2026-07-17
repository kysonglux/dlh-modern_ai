#!/usr/bin/env python3
""" open a static paroduct category page and returns a list of dictionaries"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def scrape_products(url):
    """ open a static paroduct category page and returns a list of dictionaries"""

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)

    driver.get(url)
    time.sleep(1)

    products = []

    items = driver.find_elements(By.CSS_SELECTOR, ".product, .item, .thumbnail, div")

    for item in items:
        try:
            a_tag = item.find_element(By.CSS_SELECTOR, "a[title]")
            title = a_tag.get_attribute("title")

            price = item.find_element(By.CSS_SELECTOR, "h4.price").text.strip()

            description = item.find_element(By.CSS_SELECTOR, "p.description").text.strip()

            rating_block = item.find_element(By.CSS_SELECTOR, ".ratings p[data-rating]")
            rating = rating_block.get_attribute("data-rating")

            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating
            })

        except Exception:
            continue

    driver.quit()
    return products 
