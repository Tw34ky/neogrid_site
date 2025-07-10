import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageEnhance, ImageFilter
import time
from concurrent.futures import ThreadPoolExecutor
import os


def process_page(img):
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    return pytesseract.image_to_string(img, lang='ru')


pytesseract.pytesseract.tesseract_cmd = rf'{os.path.abspath("Tesseract/tesseract.exe")}'
def pdf_to_text(doc):
    start_time = time.time()
    images = convert_from_path(doc, dpi=300)
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(process_page, images))
        print("--- %s seconds ---" % (time.time() - start_time))
        return "\n".join(texts)

