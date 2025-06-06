import streamlit as st
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from mistral import MistralClient

# Download NLTK stopwords
nltk.download('stopwords')

# Job Descriptions
job_descriptions = {
    "Data Scientist": "python machine learning pandas numpy statistics data analysis SQL sklearn regression clustering matplotlib seaborn data cleaning",
    "AI/ML Engineer": "deep learning neural networks keras tensorflow computer vision nlp CNN RNN pytorch transformers model deployment",
    "Data Analyst": "excel python data visualization SQL matplotlib seaborn tableau powerbi pandas business insights A/B testing",
    "Software Engineer": "java python c++ software design algorithms data structures system design object oriented programming Git",
    "NLP Engineer": "NLP transformers text classification BERT LLMs tokenization sentiment analysis named entity recognition spaCy",
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("🔍 Resume Analyzer (with Mistral AI)")
st.write("Upload your resume in PDF format, and get feedback powered by Mistral.")

# API Key input
mistral_api_key = st.text_input("Enter your Mistral API Key", type="password")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and mistral_api_key:
    resume_text = extract_text(uploaded_file)
    st.subheader("📝 Extracted Resume Text")
    st.write(resume_text)

    cleaned_resume = clean_text(resume_text)
    st.subheader("🧼 Cleaned Resume Text")
    st.write(cleaned_resume)

    job_title = st.selectbox("Select Job Title", list(job_descriptions.keys()))
    job_description = job_descriptions[job_title]
    cleaned_job_description = clean_text(job_description)

    # Cosine Similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_resume, cleaned_job_description])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])

    st.subheader(f"📊 Cosine Similarity with {job_title}")
    st.write(f"**Similarity Score:** `{cosine_sim[0][0]:.4f}`")

    plt.figure(figsize=(6, 3))
    plt.bar(["Resume", "JD Match"], [cosine_sim[0][0], 1 - cosine_sim[0][0]], color=['green', 'gray'])
    plt.title("Cosine Similarity Match")
    st.pyplot(plt)

    # ----- Mistral AI Feedback -----
    with st.spinner("🤖 Generating AI suggestions..."):
        prompt = f"""
You are a resume assistant. The user is applying for the role of {job_title}.
Resume:
{resume_text}

Job Description:
{job_description}

Provide:
1. Missing skills or keywords
2. 3–5 resume improvement tips
3. Optionally rewrite the resume summary
"""

        try:
            mistral_client = MistralClient(api_key=mistral_api_key)
            response = mistral_client.chat(
                model="mistral-medium",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            ai_feedback = response.choices[0].message.content
            st.subheader("🧠 Mistral AI Suggestions")
            st.write(ai_feedback)

        except Exception as e:
            st.error(f"Error calling Mistral API: {e}")
