## Student Dropout Prediction

Predicts whether a higher-education student will graduate, drop out, or remain enrolled** using a Random Forest classifier trained on a 4,424-student dataset with demographic, financial, and academic features.

## Key Findings

- Tuition payment is the strongest dropout signal - students with overdue fees drop out at 86.6% vs 24.7% for those current on payments.
- Scholarships cut dropout by 3x - holders drop out at 12.2% compared to 38.7% for non-recipients.
- 1st-semester grades dominate feature importance, the top 4 predictors are academic metrics from semesters 1-2 (14.2%, 10.9%, 9.2%, 6.0% importance), making early intervention possible within 4 months of enrollment.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Tkinter prediction GUI
python app.py
```

The GUI lets you enter a student's age, grades, scholarship status, debt, and tuition status to get a live dropout/graduate/enrolled prediction.

## Notebook

Full analysis (EDA → preprocessing → model comparison → evaluation): [`notebook/student_dropout_prediction.ipynb`](notebook/student_dropout_prediction.ipynb)
