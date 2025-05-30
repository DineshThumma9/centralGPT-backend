# import pytesseract
# import re
#
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# # Preprocess + OCR (from previous block)
# # image -> img (PIL Image)
# # text = pytesseract.image_to_string(img)
#
# # Simple fallback without preprocessing
# from PIL import Image
# img = Image.open("C:\\Users\\dines\\Projects\\adc_def\\abc\\src\\data\\img3.jpg")
# text = pytesseract.image_to_string(img)
#
# # Split by sections
# parts = re.split(r"Essay Questions", text, maxsplit=1)
# saq_raw, laq_raw = parts if len(parts) == 2 else (text, "")
#
# # Split into individual questions
# saq = re.split(r"\s*\d+\.", saq_raw)
# laq = re.split(r"\s*\d+\.", laq_raw)
#
# # Clean & remove blanks
# saq = [q.strip() for q in saq if q.strip()]
# laq = [q.strip() for q in laq if q.strip()]
#
# print("Short Answer Questions:\n")
# for i, question in enumerate(saq, 1):
#     print(f"{i}. {question}")
#
# print("\nEssay Questions:\n")
# for i, question in enumerate(laq, 1):
#     print(f"{i}. {question}")
