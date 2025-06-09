import streamlit as st
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords



nltk.download('stopwords')


job_descriptions = {
    "Data Scientist": "python machine learning pandas numpy statistics data analysis SQL sklearn regression clustering matplotlib seaborn data cleaning",
    "AI/ML Engineer": "deep learning neural networks keras tensorflow computer vision nlp CNN RNN pytorch transformers model deployment",
    "Data Analyst": "excel python data visualization SQL matplotlib seaborn tableau powerbi pandas business insights A/B testing",
    "Software Engineer": "java python c++ software design algorithms data structures system design object oriented programming Git",
    "NLP Engineer": "NLP transformers text classification BERT LLMs tokenization sentiment analysis named entity recognition spaCy",
}

# Function to clean text
def clean_text(text):
    text=text.lower()
    text=re.sub(r'\d+', '', text)  # Remove numbers
    text=re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text=re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text-' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

def extract_text(umloaded_file):
    doc=fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text=""
    for page in doc:
        text += page.get_text()
    return text

st.title("Resume Analyzer")
st.write("Upload your resume in PDF format to analyze it against job descriptions.")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    st.subheader("Extracted Resume Text")
    st.write(resume_text)

    cleaned_resume = clean_text(resume_text)
    st.subheader("Cleaned Resume Text")
    st.write(cleaned_resume)

    job_title = st.selectbox("Select Job Title", list(job_descriptions.keys()))
    job_description = job_descriptions[job_title]
    
    cleaned_job_description = clean_text(job_description)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_resume, cleaned_job_description])
    
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    
    st.subheader(f"Cosine Similarity with {job_title} Description")
    st.write(f"Similarity Score: {cosine_sim[0][0]:.4f}")
    
    plt.figure(figsize=(8, 4))
    plt.bar(["Resume", "Job Description"], [cosine_sim[0][0], 1 - cosine_sim[0][0]], color=['blue', 'orange'])
    plt.title("Cosine Similarity Visualization")
    plt.ylabel("Similarity Score")
    st.pyplot(plt)
