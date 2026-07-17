#!/usr/bin/env python3
""" open a static paroduct category page and returns dictionaries"""

import time
from selenium import webdriver


def scrape_products(url):
    """ open a static paroduct category page and returns dictionaries"""

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)

    driver.get(url)
    time.sleep(1)

    products = []

    items = driver.find_elements("css selector", "div.thumbnail")

    for item in items:
        try:
            title = item.find_element(
                "css selector", "a.title"
                ).get_attribute("title")

            price = item.find_element("css selector", "h4.price").text

            description = item.find_element(
                "css selector", "p.description").text

            rating = int(item.find_element(
                "css selector", ".ratings p[data-rating]"
                ).get_attribute("data-rating"))

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
