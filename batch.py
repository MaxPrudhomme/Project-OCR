import os
import cv2
from PIL import Image
import numpy as np
import torch

from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor


def minimal_preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened

def ocr_with_surya(image_path, langs=["en"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path)

    det_processor, det_model = load_det_processor(), load_det_model().to(device).to(torch.float32)
    rec_model, rec_processor = load_rec_model().to(device).to(torch.float32), load_rec_processor()

    predictions = run_ocr([image], langs, det_model, det_processor, rec_model, rec_processor)

    structured_text = ""
    for ocr_result in predictions:
        for text_line in ocr_result.text_lines:
            structured_text += f"{text_line.text}\n"

    return structured_text

def process_image(image_path, input_folder="batch", output_folder="results"):
    input_image_path = os.path.join(input_folder, image_path)
    output_text_path = os.path.join(output_folder, image_path.replace(".jpg", ".txt"))

    cv2.imwrite('batch/preprocessed_image.png', minimal_preprocess(input_image_path))

    # Perform OCR
    extracted_text = ocr_with_surya(input_image_path)

    # Save the extracted text to a file
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

input_folder = "batch"
output_folder = "results"

os.makedirs(output_folder, exist_ok=True)

for image_path in os.listdir(input_folder):
    process_image(image_path, input_folder, output_folder)