# CardioCheck

CardioCheck is a cardiovascular disease risk prediction project that combines a machine learning model with a lightweight web interface to help users assess potential heart-health risk based on basic medical indicators.

This project uses patient data such as age, blood pressure, BMI, cholesterol, glucose, smoking status, alcohol use, and physical activity to predict whether someone may be at risk of cardiovascular disease.

## Features

- Machine learning-based cardiovascular risk prediction
- Front-end health form for entering patient details
- Real-time prediction through a Python HTTP server
- Risk categorization: low, medium, or high
- Model accuracy display and explainable risk factors
- Clean dataset preprocessing and exploratory analysis

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- HTML / CSS / JavaScript

## Project Structure

```text
.
├── cardio_train.csv          # Training dataset
├── cardio-relief.html        # Health and care information page
├── contact-doctor.html       # Contact page
├── index.html                # Main web interface
├── main.py                   # Model training and analysis script
├── order-medicine.html       # Medicine ordering page
├── render.yaml               # Deployment configuration
├── requirements.txt          # Python dependencies
├── server.py                 # Prediction API and web server
└── cardio_graphs/            # Generated model/EDA charts
```

## How It Works

1. The project loads the cardiovascular dataset and performs preprocessing.
2. It cleans the data, engineers BMI, and removes invalid or inconsistent records.
3. It trains and compares multiple machine learning models.
4. The best-performing model is selected for prediction.
5. A web form submits patient data to the prediction API.
6. The server responds with a risk classification and contributing factors.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cardiocheck.git
cd cardiocheck
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Start the web server:

```bash
python server.py
```

Then open the app in your browser:

```text
http://localhost:8000
```

To run the training and analysis script separately:

```bash
python main.py
```

## API Endpoint

The server exposes a prediction API at:

```text
POST /predict
```

Example request body:

```json
{
  "age": 52,
  "gender": 1,
  "height": 168,
  "weight": 74,
  "ap_hi": 140,
  "ap_lo": 90,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 1,
  "alco": 0,
  "active": 0
}
```

## Model Notes

The project evaluates models such as:

- Logistic Regression
- Linear SVM

The best-performing classifier is selected automatically for prediction.

## Deployment

This project includes a Render deployment configuration in [render.yaml](render.yaml), making it suitable for quick deployment to Render.

## Notes

This project is intended for learning, demonstration, and educational healthcare analytics purposes. It should not be used as a substitute for professional medical advice.

## Future Improvements

- Add user authentication
- Improve model explainability
- Add a dashboard with charts and patient history
- Expand to more health-risk factors
- Improve UI/UX and responsive design
