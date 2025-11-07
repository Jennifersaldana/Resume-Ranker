# Resume-Ranker:
AI Resume Ranker is a Streamlit web app that lets users can upload 1–5 resumes and a job description PDF, with an optional text box to provide custom ranking criteria for the LLM (e.g., “rank by GPA” or “prioritize LSU students”), and the results are then displayed directly in the Streamlit interface. It analyzes how well each resume matches the job, identifies missing skills, and provides personalized suggestions for improvement.

## Idea:
User uploads PDF  →  We extract the text  →  Send that text to a LLM  →  Get feedback  →  Display results


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

4. Activate python environment
```Bash
source venv/bin/activate
```

5. cd into resume_analyzer 
```Bash
cd resume_analyzer 
```

6. install requirements
```Bash
pip install -r requirements.txt
```

7. View Streamlit app in your browser 

```Bash
streamlit run app.py
```


## Structure:

```Bash
Resume-Ranker/
└── resume_analyzer
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── utils/
        ├── pdf_reader.py
        └── analyzer.py   (optional – for future LLM integration)
```

## File Descriptions:

1. app.py
- The main Streamlit application that creates the web interface.

2. utils/pdf_reader.py
- A utility file that handles PDF text extraction.
- Uses the PyPDF2 library to read uploaded PDF files
- Iterates through each page and extracts the text
- Returns clean text that can be passed to the analysis logic

3. utils/analyzer.py
- A helper file for connecting to an LLM (like OpenAI GPT).
- Sends resume and job description text to the AI model
- Receives structured feedback (match score, strengths, weaknesses, suggestions)
- Returns that feedback to be displayed in app.py

4. requirements.txt
- Lists all Python dependencies needed to run the app.
