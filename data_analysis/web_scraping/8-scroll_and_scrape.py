#!/usr/bin/env python3
"""scrolls and extracts all products from a JS‐rendered infinite‐scroll page"""

import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """scrolls and extracts all products"""

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        # Scroll until page height no longer increases
        last_height = driver.execute_script(
            "return document.body.scrollHeight")

        last_count = 0

        while True:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            new_height = driver.execute_script(
                "return document.body.scrollHeight")
            count = len(driver.find_elements("css selector", "div.thumbnail"))
            if new_height == last_height and count == last_count:
                break
            last_height = new_height
            last_count = count

        products = []
        seen = set()

        cards = driver.find_elements("css selector", "div.thumbnail")

        for card in cards:
            try:
                title = card.find_element(
                    "css selector", "a.title").get_attribute("title").strip()
                price = card.find_element(
                    "css selector", "h4.price").text.strip()
                description = card.find_element(
                    "css selector", "p.description").text.strip()
                rating = len(
                    card.find_elements(
                        "css selector",
                        ".ratings .ws-icon.ws-icon-star"
                    )
                )

                key = (title, price)
                if key not in seen:
                    seen.add(key)
                    products.append({
                        "title": title,
                        "price": price,
                        "description": description,
                        "rating": rating
                    })

            except Exception:
                # Skip malformed product cards
                continue

        return products

    finally:
        driver.quit()
