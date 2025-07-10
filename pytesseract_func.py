import pytesseract
from PIL import Image
import os
import fitz


def pdf_to_text(pdf_path):
    pytesseract.pytesseract.tesseract_cmd = rf'{os.path.abspath("Tesseract/tesseract.exe")}'
    doc = fitz.open(pdf_path)
    text = ''
    for i in range(len(doc)):
        page = doc.load_page(i)
        pixmap = page.get_pixmap()
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        text = pytesseract.image_to_string(image, config='--oem 1 --psm 3')

    return text
