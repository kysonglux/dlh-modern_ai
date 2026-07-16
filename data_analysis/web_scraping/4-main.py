#!/usr/bin/env python3
# Test 2: A dummy HTML page containing JSON-LD
def fake_fetch(url, headers=None, timeout=10):
    return '''
    <html><head><script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "Quote",
        "text": "A witty quote",
        "author": {"@type": "Person","name": "Tester"},
        "keywords": ["fun","test"]
    }
    </script></head><body></body></html>
    '''

extract_jsonld_module = __import__('4-extract_jsonld')
original_fetch = extract_jsonld_module.fetch_html
extract_jsonld_module.fetch_html = fake_fetch

try:
    data = extract_jsonld_module.extract_jsonld("dummy-url")
    print(data[0])
finally:
    extract_jsonld_module.fetch_html = original_fetch