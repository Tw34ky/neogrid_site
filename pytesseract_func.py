import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import os
import fitz
from global_vars import *


def pdf_to_text(pdf_path):
    pytesseract.pytesseract.tesseract_cmd = pytesseract_exe_path
    os.environ["TESSDATA_PREFIX"] = tessdata_path
    text = ''
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pixmap = page.get_pixmap()
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            if pixmap.width * pixmap.height > 2073600:
                image.thumbnail((pixmap.width * 0.8, pixmap.height * 0.8))
            image = image.filter(ImageFilter.MedianFilter())
            image = image.convert("L")
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)
            temp = pytesseract.image_to_string(image, config=r'--oem 1 --psm 3', lang='eng+rus')
            text += temp
    return text
