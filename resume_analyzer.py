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
    "Web Developer": "HTML CSS JavaScript React Node.js frontend backend REST APIs authentication Express MongoDB responsive design Git",
    "Android Developer": "java kotlin android studio mobile app firebase XML android sdk UI UX push notifications",
    "Data Analyst": "excel python data visualization SQL matplotlib seaborn tableau powerbi pandas business insights A/B testing",
    "AI/ML Engineer": "deep learning neural networks keras tensorflow computer vision nlp CNN RNN pytorch transformers model deployment",
    "Backend Developer": "Node.js Express REST APIs MongoDB MySQL PostgreSQL authentication JWT cloud server Git Docker",
    "Frontend Developer": "HTML CSS JavaScript React Vue.js responsive design accessibility Tailwind Bootstrap Figma UI",
    "DevOps Engineer": "AWS Docker Kubernetes CI/CD Jenkins Terraform Linux Bash Git version control monitoring scalability",
    "Cybersecurity Analyst": "network security penetration testing firewalls ethical hacking Kali Linux threat analysis incident response SIEM",
    "Software Engineer": "java python c++ software design algorithms data structures system design object oriented programming Git",
    "Cloud Engineer": "AWS Azure GCP cloud computing EC2 S3 Lambda Kubernetes Terraform DevOps cloud architecture monitoring",
    "Full Stack Developer": "HTML CSS JavaScript React Node.js Express MongoDB SQL Git REST APIs deployment authentication",
    "Business Analyst": "excel SQL data visualization tableau powerbi business strategy modeling documentation communication Agile",
    "Product Manager": "product strategy user stories backlog roadmap wireframes market research analytics agile stakeholder management",
    "UI/UX Designer": "Figma Adobe XD wireframes prototypes user flows usability testing interaction design accessibility branding",
    "NLP Engineer": "NLP transformers text classification BERT LLMs tokenization sentiment analysis named entity recognition spaCy",
    "Computer Vision Engineer": "OpenCV image processing deep learning CNN object detection image classification segmentation YOLO",
    "Game Developer": "Unity Unreal Engine C# C++ game physics animation game design mobile games 2D 3D graphics",
    "Blockchain Developer": "solidity ethereum smart contracts web3.js dApps crypto blockchain security consensus tokens",
    "QA Tester": "manual testing automation selenium JUnit test cases regression testing bug tracking SDLC test scripts",
}

def clean_text(text):
    text=text.lower()
    text=re.sub(r'\d+', '', text)  # Remove numbers
    text=re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text=re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
  # Remove stopwords
    return text

def extract_text(umloaded_file):
    doc=fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text=""
    for page in doc:
        text+=page.get_text()
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
