# Placement Predictor

A basic Flask website for your CGPA/IQ placement prediction model.

## What's inside

- `app.py` — Flask backend that loads the model and serves the form
- `templates/index.html` — the page (form + result)
- `static/style.css` — basic styling
- `model.pkl` — your trained LogisticRegression model
- `scaler.pkl` — the StandardScaler used to scale CGPA/IQ before prediction
- `requirements.txt` — Python packages needed

## How to run it

1. Open a terminal in this folder.
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   python app.py
   ```
4. Open your browser to: http://127.0.0.1:5000

Enter a CGPA and IQ, hit Predict, and it'll tell you "Ho jayega Placement!" or "Beta Tumse na ho payega!" along with the model's confidence.

## Important note about scaler.pkl

Your original notebook fit a `StandardScaler` on the training data but never saved it with `pickle.dump`, only the model was saved. Without it, the model gets fed raw CGPA/IQ values and basically always predicts "Placement" no matter what's entered, since the model was trained on scaled values, not raw ones.

I rebuilt a `scaler.pkl` by fitting a fresh `StandardScaler` on your full `placement.csv` dataset (same columns: cgpa, iq). I checked it against your model: it gets 92% accuracy on the full dataset and produces sensible, varied predictions across realistic CGPA/IQ values. It's a very close match to the original scaler, but not pixel-identical (the original was fit on a random 90-row train split, not all 100 rows), so confidence percentages near the decision boundary might shift very slightly from your original notebook's results. If you still have the original notebook's exact `X_train` split or a saved `scaler.pkl`, swap it in for an exact match.

## Notes

- This is intentionally simple, no database, no deployment config, just a local Flask app you can run and demo.
- This model is made up of very small dataset (Toy_data) just for Learning Purspose
