# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
#
#
# def pdf_to_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#
#     for page in doc:
#         # Get the page as an image
#         pix = page.get_pixmap(dpi=300)  # 300 DPI for good OCR
#         img = Image.open(io.BytesIO(pix.tobytes()))
#
#         # Use Tesseract OCR
#         text += pytesseract.image_to_string(img)
#
#     return text
#
#
# # Usage
# text = pdf_to_text("scanned.pdf")
# print(text)