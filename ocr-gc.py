import os
from PIL import Image
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

PROJECT_ID = ""
LOCATION = ""
PROCESSOR_ID = ""
MIME_TYPE = "image/jpeg"
INPUT_FOLDER = "batch/source"
OUTPUT_FOLDER = "batch/resultsGCP"

Image.MAX_IMAGE_PIXELS = None

docai_client = documentai.DocumentProcessorServiceClient(
    client_options=ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
)

RESOURCE_NAME = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

with open('batch/editor.txt', 'r', encoding='utf-8') as editor_file:
    editor_content = editor_file.read()

def ocr(file_path):
    with open(file_path, "rb") as image:
        image_content = image.read()

    raw_document = documentai.RawDocument(content=image_content, mime_type=MIME_TYPE)

    request = documentai.ProcessRequest(name=RESOURCE_NAME, raw_document=raw_document)

    result = docai_client.process_document(request=request)

    output_file_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path) + '.txt')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(editor_content + '\n' + result.document.text)

def process_batch(input_folder, count = 0):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for file_name in os.listdir(input_folder):
        if count == 50:
            break
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
            ocr(file_path)
        count += 1

def preprocess(directory, width_threshold=2200, height_threshold=2910):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                try:
                    file_path = os.path.join(root, file)
                    
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        if width > width_threshold or height > height_threshold:
                            aspect_ratio = min(width_threshold / width, height_threshold / height)
                            new_size = (int(width * aspect_ratio), int(height * aspect_ratio))
                            
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                            img.save(file_path)
                            
                            print(f"Resized {file} to {new_size}")
                        else:
                            print(f"No resize needed for {file}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    preprocess('batch/source')
    process_batch(INPUT_FOLDER)
