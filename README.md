# Resume-Ranker
Project 2 for CSC 4444 - AI

Team:
1. Jennifer Saldana
2. Sadikshya Gyawali
3. Kshitiz Dhungana
4. Candor Alemu
5. Anastacia Muhammad
6. Eli Rayburn 

# Resume-Ranker:
AI Resume Ranker is a Streamlit web app that lets users can upload 1–5 resumes and a job description PDF, and the results are then displayed directly in the Streamlit interface. It analyzes how well each resume matches the job, identifies missing skills, and provides personalized suggestions for improvement using a local Ollama3

## Idea:
User uploads PDF  →  We extract the text  →  Send that text to a model  →  Display results →  Get personalized feedback  


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
    └── requirements.txt
└── venv
└── .gitignore
└── README.md
```

## File Descriptions:

1. app.py
- The main Streamlit application that creates the web interface.

2. requirements.txt
- Lists all Python dependencies needed to run the app.

3. best_matching_model.pkl
- Fine Tuned Model.

## Folder Descriptions:
1. resumes-job
- One job description called "[IT_manager_job_description](resumes-job/IT_manager_job_description.pdf)"
- Many security/analyst roles
