# Resume-Ranker
Project 2 for CSC 4444 - AI
# Resume-Ranker:
AI Resume Ranker is a Streamlit web app that lets users can upload 1–5 resumes and a job description PDF, with an optional text box to provide custom ranking criteria for the model (e.g., “rank by GPA” or “prioritize LSU students”), and the results are then displayed directly in the Streamlit interface. It analyzes how well each resume matches the job, identifies missing skills, and provides personalized suggestions for improvement.

## Idea:
User uploads PDF  →  We extract the text  →  Send that text to a model  →  Get feedback  →  Display results


## How to Run:
1. Clone or download this repository

```bash
git clone https://github.com/yourusername/Resume-Ranker.git
```

2. cd into Project Folder
```Bash
cd Resume-Ranker
```

3. Set up python environment
```Bash
python3 -m venv venv
```

OR
```Bash
python -m venv venv
```

4. Activate python environment

MacOS:
```Bash 
source venv/bin/activate
```
Windows:
```Bash 
venv\Scripts\activate
```

5. cd into resume_analyzer 
```Bash
cd streamlit
```

6. install requirements
```Bash
pip install -r requirements.txt
```

7. View Streamlit app in your browser 

```Bash
streamlit run app.py
```

*Press Enter to skip intro.*

8. ollama download
- download https://ollama.com/download/mac
- ollama pull llama3
- ollama run llama3



## Structure:

```Bash
Resume-Ranker/
└── resume_job
└── streamlit
    │
    ├── app.py
    ├── best_matching_model.pkl
    ├── requirements.txt
└── venv
└── .gitignore
└── README.md
```

## File Descriptions:

1. app.py
- The main Streamlit application that creates the web interface.

2. requirements.txt
- Lists all Python dependencies needed to run the app.