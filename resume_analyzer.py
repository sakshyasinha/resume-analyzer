import streamlit as st
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Expanded job descriptions
job_descriptions = {
    "Data Scientist": "python machine learning pandas numpy statistics data analysis SQL sklearn regression clustering matplotlib seaborn data cleaning",
    "AI/ML Engineer": "deep learning neural networks keras tensorflow computer vision nlp CNN RNN pytorch transformers model deployment",
    "Data Analyst": "excel python data visualization SQL matplotlib seaborn tableau powerbi pandas business insights A/B testing",
    "Software Engineer": "java python c++ software design algorithms data structures system design object oriented programming Git",
    "NLP Engineer": "NLP transformers text classification BERT LLMs tokenization sentiment analysis named entity recognition spaCy",
    "DevOps Engineer": "CI/CD docker kubernetes aws azure cloud infrastructure monitoring linux jenkins terraform automation",
    "Backend Developer": "python java nodejs API REST SQL databases scalability security performance flask django express",
    "Frontend Developer": "html css javascript react vue angular UI UX responsive design accessibility web performance testing",
    "Cybersecurity Analyst": "network security encryption firewalls risk assessment vulnerability scanning SIEM penetration testing",
    "Product Manager": "product roadmap user stories agile scrum stakeholder communication market research feature prioritization metrics"
}

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to extract text from PDF
def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.title("📄 Resume Analyzer")
st.write("Upload your resume in PDF format to analyze how well it matches selected job descriptions.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF only)", type="pdf")

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    st.subheader("📝 Extracted Resume Text")
    st.write(resume_text)

    cleaned_resume = clean_text(resume_text)
    st.subheader("🧹 Cleaned Resume Text")
    st.write(cleaned_resume)

    job_title = st.selectbox("🎯 Select Job Title", list(job_descriptions.keys()))
    job_description = job_descriptions[job_title]
    cleaned_job_description = clean_text(job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_resume, cleaned_job_description])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    st.subheader(f"📊 Match Score for: {job_title}")
    st.markdown(f"**Similarity Score:** `{cosine_sim:.4f}`")

    if cosine_sim > 0.7:
        st.success("✅ Great match! Your resume closely aligns with the job description.")
    elif cosine_sim > 0.4:
        st.warning("⚠️ Moderate match. Consider tailoring your resume further to this role.")
    else:
        st.error("❌ Low match. You may need to better align your resume with the job description.")
