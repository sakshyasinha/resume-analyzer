import streamlit as st
import fitz  
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


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
    "Product Manager": "product roadmap user stories agile scrum stakeholder communication market research feature prioritization metrics",
    
    "Machine Learning Engineer": "machine learning algorithms model training feature engineering data preprocessing deployment performance tuning",
    "Data Engineer": "ETL data pipelines big data architecture SQL NoSQL data warehousing Apache Spark Hadoop data modeling",
    "Cloud Architect": "cloud computing architecture AWS Azure GCP microservices serverless architecture security compliance",
    "Business Intelligence Analyst": "data visualization BI tools SQL data modeling reporting dashboards insights business strategy",
    "Game Developer": "game design C# Unity Unreal Engine gameplay mechanics graphics programming AI in games",
    "Mobile Developer": "iOS Android Swift Kotlin mobile app development UI/UX design cross-platform frameworks React Native Flutter",
    "Systems Administrator": "system administration network management troubleshooting security backups virtualization scripting",
    "Technical Writer": "documentation technical writing user manuals API documentation content creation communication skills",
    "Research Scientist": "research methodology data analysis experimental design statistical modeling scientific writing",
    "Sales Engineer": "technical sales product demonstrations customer engagement solution selling relationship management",
    
    "Full Stack Developer": "javascript python nodejs react angular vue backend frontend REST APIs databases DevOps",
    "Quality Assurance Engineer": "testing automation selenium manual testing test cases bug tracking continuous integration",
    "UX Designer": "user experience wireframing prototyping usability testing user research interaction design visual design",
    "Data Architect": "data modeling database design big data governance metadata management ETL pipelines cloud data lakes",
    "Site Reliability Engineer": "monitoring incident response automation scalability uptime Kubernetes Prometheus Grafana chaos engineering",
    "Blockchain Developer": "blockchain smart contracts solidity ethereum cryptography decentralized applications NFT DeFi",
    "AR/VR Developer": "augmented reality virtual reality 3D modeling Unity Unreal depth sensors spatial computing",
    "IT Support Specialist": "technical support troubleshooting help desk hardware software ticketing systems customer service",
    "Scrum Master": "agile scrum ceremonies facilitation team coaching backlog management sprint planning collaboration",
    "Instructional Designer": "e-learning curriculum development training materials instructional technology multimedia design",
    
    "Cloud Engineer": "cloud infrastructure AWS Azure GCP automation security networking cost optimization",
    "Mobile QA Engineer": "mobile testing automation Espresso XCTest Appium device labs performance testing crash analytics",
    "Data Visualization Specialist": "D3.js Tableau Power BI charts dashboards storytelling data communication UX",
    "Embedded Systems Engineer": "C C++ microcontrollers real-time operating systems hardware interfacing IoT firmware development",
    "Information Security Manager": "risk management compliance policy enforcement incident response governance audits",
    "Marketing Analyst": "market research analytics data segmentation campaign measurement digital marketing SEO SEM",
    "DevSecOps Engineer": "security automation pipeline integration vulnerability scanning compliance IaC cloud security",
    "Localization Specialist": "translation internationalization cultural adaptation language testing global software",
    "CRM Administrator": "customer relationship management Salesforce HubSpot data integration workflows reporting",
    "Database Administrator": "SQL tuning backup recovery replication performance security access management",
    
    "Artificial Intelligence Researcher": "AI algorithms reinforcement learning generative models robotics natural language processing cognitive computing",
    "Systems Engineer": "systems integration network design performance tuning server management security architectures",
    "Web Developer": "HTML CSS JavaScript responsive design SEO accessibility cross-browser compatibility front-end frameworks",
    "Technical Program Manager": "project management technical delivery cross-functional coordination resource planning risk management",
    "Content Strategist": "content planning creation distribution SEO analytics storytelling brand voice",
    "Ethical Hacker": "penetration testing vulnerability analysis exploit development social engineering network protocols",
    "Software Tester": "test planning execution defect reporting automation frameworks regression testing quality standards",
    "Audio Engineer": "sound editing mixing mastering live sound recording acoustics signal processing",
    "Video Game Designer": "game mechanics storytelling level design player engagement prototyping user feedback iterations",
    "IoT Developer": "internet of things embedded devices sensors protocols edge computing security data streaming"
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

    # Identify missing keywords
    job_keywords = set(cleaned_job_description.split())
    resume_keywords = set(cleaned_resume.split())
    missing_keywords = job_keywords - resume_keywords

    if missing_keywords:
        st.subheader("🔍 Missing Keywords")
        st.write("Consider adding the following keywords to improve your match:")
        st.write(", ".join(missing_keywords))
    else:
        st.success("✅ All relevant keywords are present in your resume!")
