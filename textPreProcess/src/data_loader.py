from docx import Document

# Load the Word file
def extract_word_text(path):
    # file_path = "Bishal Shrestha.docx"
    doc = Document(path)  # replace with your file path

    # Extract all text
    full_text = []

    for para in doc.paragraphs:
        full_text.append(para.text)

    text = "\n".join(full_text)
    # print(text)
    return text