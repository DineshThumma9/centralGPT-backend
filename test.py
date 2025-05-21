from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


img = Image.open("img.png")

print(pytesseract.image_to_string(img))

print("-----------------------------------------------------------------")

print(pytesseract.image_to_data(img))

print("-------------------------------------------------------------------")

print(pytesseract.image_to_boxes(img))