import cv2
from PIL import Image
import numpy as np

def minimal_preprocess(image_path): # Works fine for image with great scan quality
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    return sharpened

def rotation_preprocess(image_path): # Works only on rotated text
    img = cv2.imread(image_path)
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Use Canny edge detection to find the edges of text
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Step 3: Use Hough Line Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Step 4: Check for lines close to 90 degrees (rotated text)
    rotated_img = img.copy()
    angle_correction_needed = False
    if lines is not None:
        for rho, theta in lines[:, 0]:
            degree = np.degrees(theta)
            if 80 <= degree <= 100:  # Detect lines around 90 degrees
                angle_correction_needed = True
                break
    
    # Step 5: If rotated content is detected, apply 90-degree correction
    if angle_correction_needed:
        # Rotate the image by 90 degrees clockwise
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    # Step 6: Process the (possibly rotated) image
    gray_rotated = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_rotated, (3, 3), 0)
    sharpened = cv2.addWeighted(gray_rotated, 1.5, blurred, -0.5, 0)
    
    return sharpened


folder = 'testA/'
image_path = 'image.jpg'

preprocessed_image = minimal_preprocess(folder + image_path)
preprocessed_image_rotated = rotation_preprocess(folder + image_path)

cv2.imwrite(folder + 'preprocessed_image.png', preprocessed_image)
cv2.imwrite(folder + 'preprocessed_image_rotation.png', preprocessed_image)