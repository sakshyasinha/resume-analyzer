
# 📄 Resume Analyzer

**🔗 Live Demo:** [Resume Analyzer App](https://sakshyasinha-resume-analyzer-resume-analyzer-d6gj8f.streamlit.app/)

A Streamlit-powered NLP tool that analyzes your resume and measures how well it matches a selected job description using **TF-IDF** and **Cosine Similarity**. Perfect for job seekers aiming to improve their resume targeting.

---

## 🔍 Features

- 📤 Upload your resume in **PDF** format
- 📝 Extracts and displays resume text
- 🎯 Choose from **50+ predefined job titles**
- 📊 Calculates a **match score** against the job description
- 🧠 Identifies **missing keywords** to improve alignment
- ✅ Simple and responsive user interface using Streamlit

---

## 🚀 Technologies Used

- **Streamlit** – for building the interactive web app
- **PyMuPDF (fitz)** – for PDF text extraction
- **NLTK** – for stopword removal
- **scikit-learn** – for TF-IDF vectorization and similarity scoring

---

## 📁 Project Structure



resume-analyzer/
├── main.py                # Streamlit application code
├── requirements.txt       # Python dependencies
└── README.md              # Project description


## 🧠 How It Works

1. Upload a .pdf resume.
2. The app extracts and cleans the text.
3. Select a job title from the dropdown.
4. The app vectorizes both resume and job description text using TF-IDF.
5. Cosine similarity calculates how closely they match.
6. Missing keywords are listed to help you tailor your resume.



## 🛠️ Installation

To run this app locally:

1. Clone the repository:

bash
git clone https://github.com/your-username/resume-analyzer.git
cd resume-analyzer


2. (Optional) Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the dependencies:

bash
pip install -r requirements.txt


4. Run the Streamlit app:

bash
streamlit run main.py


---

## 📌 Example Use Case

* **Selected Role:** Data Scientist
* **Resume Match Score:** `0.0989`
* **Suggested Keywords:** seaborn, statistics, pandas, SQL, numpy, analysis, cleaning, matplotlib, sklearn

---

## 🧾 Job Titles Included

* Data Scientist
* AI/ML Engineer
* Data Analyst
* Software Engineer
* Backend Developer
* Frontend Developer
* Full Stack Developer
* Cybersecurity Analyst
* UX Designer
* NLP Engineer
  ...and many more (50+ roles total)

You can customize or expand the job descriptions from the job_descriptions dictionary in the code.

---





> If this project helped you, consider giving it a ⭐ on GitHub!

