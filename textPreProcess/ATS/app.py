import streamlit as st
import spacy
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")

st.title("AI Resume Match Analyzer")
st.caption("Classical NLP-based resume evaluation")

# ---------------- LOAD MODEL (CACHED) ----------------
import subprocess

@st.cache_resource
def load_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Download model if missing
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_model()

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ---------------- KEY PHRASE EXTRACTION ----------------
def extract_key_phrases(doc):
    phrases = set()

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()

        if (
            len(cleaned) > 2
            and not all(token.is_stop for token in chunk)
            and not any(token.pos_ == "PRON" for token in chunk)
        ):
            phrases.add(cleaned)

    return phrases

# ---------------- ENTITY EXTRACTION ----------------
def extract_entities(doc):
    entity_dict = {}

    for ent in doc.ents:
        if ent.label_ not in entity_dict:
            entity_dict[ent.label_] = set()
        entity_dict[ent.label_].add(ent.text)

    return entity_dict

# ---------------- SIMILARITY ----------------
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# ---------------- UI INPUT ----------------
job_desc = st.text_area("Paste Job Description")
resume_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# ---------------- ANALYSIS ----------------
if st.button("Analyze"):

    if job_desc and resume_pdf:

        # Show a spinner while processing
        with st.spinner("Analyzing resume... This may take a few seconds."):
            # Optional: small progress bar simulation
            progress_bar = st.progress(0)
            
            resume_text = extract_text_from_pdf(resume_pdf)
            progress_bar.progress(20)

            # Process text with NLP (this is the heavy part)
            resume_doc = nlp(resume_text)
            progress_bar.progress(50)

            job_doc = nlp(job_desc)
            progress_bar.progress(70)

            resume_phrases = extract_key_phrases(resume_doc)
            job_phrases = extract_key_phrases(job_doc)
            progress_bar.progress(80)

            matched = sorted(resume_phrases & job_phrases)
            missing = sorted(job_phrases - resume_phrases)

            similarity_score = compute_similarity(resume_text, job_desc)
            entities = extract_entities(resume_doc)
            progress_bar.progress(100)

        # Clear progress bar after completion
        progress_bar.empty()
        
        # ---------------- SUMMARY SECTION ----------------
        st.markdown("## 📊 Match Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Match Score", f"{similarity_score}%")
        col2.metric("Matched Skills", len(matched))
        col3.metric("Skill Gaps", len(missing))

        st.divider()

        # ---------------- SKILL BREAKDOWN ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matching Skills")
            if matched:
                for skill in matched[:15]:
                    st.markdown(f"- {skill.title()}")
            else:
                st.info("No strong matches detected.")

        with col2:
            st.markdown("### ❌ Missing Skills")
            if missing:
                for skill in missing[:15]:
                    st.markdown(f"- {skill.title()}")
            else:
                st.success("No major skill gaps identified.")

        st.divider()

        # ---------------- RESUME INSIGHTS ----------------
        st.markdown("### 📄 Resume Insights")

        if entities:
            for label, values in entities.items():
                st.markdown(f"**{label}**")
                st.write(", ".join(values))
        else:
            st.info("No significant named entities detected.")

    else:
        st.warning("Please upload a resume and paste the job description.")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <p style="text-align:center; font-size:14px; color:gray;">
        Created by <a href="https://github.com/Bishal-Stha" target="_blank">Bishal Shrestha</a>
    </p>
    """,
    unsafe_allow_html=True
)