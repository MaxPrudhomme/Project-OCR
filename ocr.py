import cv2
from PIL import Image
import numpy as np
import pytesseract

# Path to Tesseract executable (only necessary if not in system PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\accou\Documents\Tesseract\tesseract.exe'

def minimal_preprocess(image_path): # Works fine for image with great scan quality
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    return sharpened

def perform_ocr(image_path):
    preprocessed_image = minimal_preprocess(image_path)
    cv2.imwrite(folder + 'preprocessed_image.png', preprocessed_image)
    custom_config = r'--oem 3 --psm 1'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    
    return text

folder = 'testE/'

image_path = 'image.jpg'
extracted_text = perform_ocr(folder + image_path)
with open(folder + 'extracted_text.txt', 'w', encoding='utf-8') as file:
    file.write(extracted_text)

print(extracted_text)