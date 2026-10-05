# Support Ticket Classifier

A small FastAPI service that reads a support ticket and predicts its category: `login`, `network`, `payment`, or `other`.

This is a practice project to learn the basic machine learning workflow: data, training, saving a model, serving it with an API, and testing it.

## Tech Stack

- Python 3.12
- FastAPI and Uvicorn
- scikit-learn (TF-IDF and Logistic Regression)
- pandas
- joblib
- pytest

## How It Works

1. The ticket text is converted into numbers using **TF-IDF**.
2. A **Logistic Regression** model predicts the category from those numbers.
3. The trained model and the TF-IDF vectorizer are saved with `joblib`.
4. The FastAPI app loads both files and serves predictions at `POST /predict`.

## Dataset

- 100 sample tickets in `data/tickets.csv`
- 4 categories with 25 tickets each: `login`, `network`, `payment`, `other`
- All tickets are **synthetic**, written by me. No company or customer data is used.

## Project Structure

```
support-ticket-classifier/
├── data/
│   └── tickets.csv
├── models/
│   ├── model.joblib
│   └── vectorizer.joblib
├── explore.py
├── train.py
├── main.py
├── test_main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/Ganesh-AK/support-ticket-classifier.git
cd support-ticket-classifier
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## Run

Train the model first. This creates the `models/` files:

```bash
python train.py
```

Start the API:

```bash
uvicorn main:app --reload
```

Open the interactive docs at http://127.0.0.1:8000/docs to try the API in your browser.

## API

### `GET /`

Health check.

```json
{"message": "Hello, Ticket Classifier"}
```

### `POST /predict`

Request:

```json
{"text": "I forgot my password and my account is locked"}
```

Response:

```json
{"category": "login"}
```

If the `text` field is missing, the API returns `422 Unprocessable Entity`.

## Results

| Metric | Value |
|---|---|
| Train / test split | 80 / 20 (stratified, `random_state=42`) |
| Accuracy on test set | 0.85 |

**Note:** The test set has only 20 tickets, so this number is a rough estimate and can change with a different split or more data.

## Tests

```bash
pytest
```

The tests check the home route, the prediction for login, payment and network tickets, and the error for a missing `text` field.

## Limitations

- The dataset is small and synthetic, so the model may not work well on real tickets.
- Some categories overlap (for example, a login problem on a new phone can look like `login` or `network`).

## Future Improvements

- Add more and more varied tickets.
- Try other models and compare them with cross-validation.
- Add a confidence score to the `/predict` response.
- Package the app with Docker.

## Author

Ganesh Arumugam