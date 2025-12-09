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
AI Resume Ranker is a Streamlit web app that lets users can upload 1–5 resumes and a job description PDF, and the results are then displayed directly in the Streamlit interface. It analyzes how well each resume matches the job, identifies missing skills, and provides personalized suggestions for improvement using a local Ollama3.

## Idea:
User uploads PDF  →  We extract the text  →  Send that text to a model  →  Display results  


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
└── polished_resumes
└── resume-job
└── streamlit
    │
    ├── best_matching_model.pkl
    └── requirements.txt
    ├── app.py
    └── features/
        ├── text_utils.py
        ├── bias_utils.py
        ├── scoring_utils.py
        ├── visuals.py
        └── llm_feedback.py
└── venv
└── .gitignore
└── ai_project.ipynb
└── README.md
└── resume_dataset.csv
```

## File Descriptions:

1. app.py
- The main Streamlit application that creates the web interface.

2. requirements.txt
- Lists all Python dependencies needed to run the app.

3. best_matching_model.pkl
- Fine Tuned Model.

4. ai_project.ipynb
- Code for trained model

5. resume_dataset.csv
- Dataset used to train model.


## Folder Descriptions:
1. /resumes-job
- One job description called "[IT_manager_job_description](resumes-job/IT_manager_job_description.pdf)"
- Many security/analyst roles

2. /polished_resumes
- Resumes that are created to test prediction scores, bias indicators, skill extraction, missing skills detection, and LLM feedback.

3. /features
- Ranking tab with scores, tiers, confidence, word count, keyword match
- Feedback tab using Ollama, personalized and tier-aware
- Skills tab showing extracted vs missing skills
- Bias tab showing heuristic bias indicators
- Metrics tab with score histogram, wordcount vs score plot, and resume similarity heatmap