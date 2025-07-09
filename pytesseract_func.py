import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor
import os


pytesseract.pytesseract.tesseract_cmd = rf'{os.path.abspath("Tesseract/tesseract.exe")}'
def pdf_to_text(doc):
    start_time = time.time()
    text = ""

    def process_page(img):
        return pytesseract.image_to_string(img)

    for page in doc:
        # Get the page as an image
        pix = page.get_pixmap(dpi=300)  # 300 DPI for good OCR
        img = Image.open(io.BytesIO(pix.tobytes()))
        with ThreadPoolExecutor() as executor:
            texts = list(executor.map(process_page, img))
        return "\n".join(texts)
    print("--- %s seconds ---" % (time.time() - start_time))
    return text
