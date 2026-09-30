#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
import pprint
import os
image_classifier = __import__('8-image_classifier').image_classifier

image_path = "dog.jpg"


classification_model = "google/vit-base-patch16-224"

classifier = image_classifier(classification_model)

image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

classification_result = classifier(image_path)
pprint.pprint({"Classification result": classification_result})