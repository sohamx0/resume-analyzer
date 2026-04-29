# Smart & Explainable Resume Analyzer

A full-stack Flask application that analyzes a resume against a job description using:

- spaCy (`en_core_web_sm`) for NLP preprocessing and candidate term extraction
- Sentence Transformers (`all-MiniLM-L6-v2`) for semantic matching
- Dataset-driven skill inventory built from category folders in `data/data/`

## Project Structure

- `app/main.py`: Flask app and API endpoints
- `app/parser.py`: Resume parsing and quality analysis
- `app/dataset_processor.py`: Dataset scanning and `data/skills.txt` generation
- `app/skills_extractor.py`: Skill extraction against generated skill inventory
- `app/matcher.py`: Semantic similarity + category prediction
- `app/scorer.py`: Weighted scoring logic
- `app/suggestions.py`: Explainable suggestion generation
- `app/templates/index.html`: Frontend page
- `app/static/styles.css`: UI styling
- `app/static/app.js`: Browser logic

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

3. Run the application:

```bash
python app/main.py
```

The server starts at `http://localhost:5000`.

## Model retraining

If you change the feature set or want calibrated confidence improvements, regenerate data and retrain:

```bash
python dataset_generator.py
python train_models.py
```

## Dataset

Expected dataset location:

- `data/data/<CATEGORY_NAME>/*.pdf|*.docx|*.txt`

On startup, if `data/skills.txt` or `data/category_profiles.json` is missing, the app runs dataset processing automatically.

You can also run processing manually:

```bash
python app/dataset_processor.py
```

## API Endpoints

- `POST /upload_resume`: Upload resume file (`resume` field in multipart form)
- `POST /analyze`: Analyze resume against job description
- `GET /results`: Get latest result or `?analysis_id=<id>`

### Example `/analyze` response

```json
{
  "score": 84,
  "similarity_score": 0.79,
  "matched_skills": ["python", "machine learning"],
  "missing_skills": ["sql", "aws"],
  "strengths": ["Strong semantic alignment with the job description."],
  "weaknesses": ["Skill overlap is low compared to job requirements."],
  "suggestions": [
    "Add evidence for missing skills in projects or experience: aws, sql.",
    "Mention tools and technologies explicitly in project and experience bullets."
  ]
}
```

## Scoring Formula

- Semantic similarity: 50%
- Skill match: 30%
- Resume quality: 20%

Overall score is mapped to `0-100`.
