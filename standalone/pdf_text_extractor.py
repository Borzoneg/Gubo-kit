from pdf2image import convert_from_path
import pytesseract

images = convert_from_path("/home/gu/Downloads/PlayersHandbook2024-239-342.pdf", dpi=300)

text = "\n".join(
    pytesseract.image_to_string(img, lang="eng")
    for img in images
)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
