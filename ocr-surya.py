import torch
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open('testA/preprocessed_image.png')
langs = ["en"]

det_processor, det_model = load_det_processor(), load_det_model().to(device).to(torch.float32)
rec_model, rec_processor = load_rec_model().to(device).to(torch.float32), load_rec_processor()

predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

structured_text = ""
for ocr_result in predictions:
    for text_line in ocr_result.text_lines:
        structured_text += f"{text_line.text}\n"

with open('testA/surya.txt', 'w', encoding='utf-8') as file:
    file.write(structured_text)