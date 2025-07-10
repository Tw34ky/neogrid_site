import easyocr
reader = easyocr.Reader(['ru'])
import fitz
import numpy as np


def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        results = reader.readtext(np.array(pix))
        text += "\n".join([res[1] for res in results])

    return text