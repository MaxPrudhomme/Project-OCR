import warnings
import cv2
import numpy as np
from PIL import Image
import easyocr
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

def minimal_preprocess(image_path):
    """Preprocess the image for better OCR results."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    return sharpened

def clean_text(text):
    """Clean the extracted text."""
    text = re.sub(r'[^\w\s.,?!:;\'"-]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

def perform_ocr(image_path):
    """Perform OCR using EasyOCR."""
    preprocessed_image = minimal_preprocess(image_path)
    preprocessed_image_path = 'preprocessed_image.png'
    cv2.imwrite(preprocessed_image_path, preprocessed_image)
    
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])  # You can add more languages if needed
    
    # Read text from the preprocessed image
    results = reader.readtext(preprocessed_image_path)
    
    # Extract text
    text = ""
    for result in results:
        text += result[1] + " "
    
    return clean_text(text)


def perform_ocr_with_layout(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path, detail=1)  # detail=1 to get bounding boxes

    # Sort results by vertical position (top-to-bottom)
    results.sort(key=lambda x: x[0][0][1])  # Sorting by the top-left y-coordinate

    structured_text = ""
    for (bbox, text, confidence) in results:
        structured_text += f"{text}\n"

    return structured_text

folder = 'testA/'
image_path = folder + 'image.jpg'
extracted_text = perform_ocr_with_layout(image_path)

with open(folder + 'extracted_text.txt', 'w', encoding='utf-8') as file:
    file.write(extracted_text)

print(extracted_text)
