# resume_analyzer.py

import streamlit as st
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# -------------------
# JOB ROLE TEMPLATES
# -------------------

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


# -------------------
# CLEAN TEXT FUNCTION
# -------------------
import matplotlib.pyplot as plt

st.subheader("📊 Match Score Chart:")
sorted_roles = sorted(zip(roles, similarities), key=lambda x: x[1], reverse=True)
top_roles, top_scores = zip(*sorted_roles)

fig, ax = plt.subplots()
ax.barh(top_roles[::-1], top_scores[::-1])
ax.set_xlabel("Match Score")
ax.set_title("Resume vs Job Role Similarity")
st.pyplot(fig)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# -------------------
# EXTRACT RESUME TEXT
# -------------------

def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------------------
# STREAMLIT UI
# -------------------

st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume and get job role recommendations!")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    st.subheader("📄 Extracted Text Preview:")
    st.text_area("", resume_text[:1000], height=300)

    clean_resume = clean_text(resume_text)

    # TF-IDF
    roles = list(job_descriptions.keys())
    corpus = list(job_descriptions.values())
    corpus.append(clean_resume)  # last one is user's resume

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)

    resume_vector = vectors[-1]
    role_vectors = vectors[:-1]

    similarities = cosine_similarity(resume_vector, role_vectors)[0]

    # Get best match
    best_idx = similarities.argmax()
    best_role = roles[best_idx]
    confidence = similarities[best_idx] * 100

    st.subheader("🔍 Best Match:")
    st.success(f"**{best_role}** ({confidence:.2f}% match)")

    st.write("📌 Other Scores:")
    for role, score in zip(roles, similarities):
        st.write(f"{role}: {score:.2f}")
