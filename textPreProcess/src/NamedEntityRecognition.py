from data_loader import extract_word_text
import spacy

nlp = spacy.load("en_core_web_trf")

# Add an EntityRuler before the NER pipeline
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {"label": "SKILL", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]},
]

ruler.add_patterns(patterns)

path = "Bishal Shrestha.docx"
text = extract_word_text(path)

doc = nlp(text)
for token in doc:
    if token.ent_type_ and not token.is_punct:
        print(f"{token} -> {token.ent_type_}")
