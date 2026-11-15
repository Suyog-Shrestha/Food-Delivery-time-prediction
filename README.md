# Food Delivery Time Prediction

Midterm project for the [DataTalks.Club Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

Delivery platforms need to tell customers how long an order will take, and they need to do it before the order leaves the restaurant. An estimate that is too short disappoints the customer. One that is too long may discourage the order altogether. It also helps the platform plan how many riders it needs.

This project predicts the delivery time of an order, in minutes, from information available when the order is placed: the rider's age and rating, the weather, the traffic level, the distance between restaurant and customer, the type of order and vehicle, and the time of day. The task is a regression problem, and the model is exposed as a web service so that other systems can request predictions.

## Dataset

The data is the Kaggle food delivery dataset, with 45,593 completed deliveries. The training file is stored at `data/train.csv`.

Each row describes one delivery. The target is the delivery time in minutes. The main input features are:

- Delivery person age and rating
- Weather conditions
- Road traffic density
- Distance between restaurant and delivery location
- Type of order and type of vehicle
- Festival and city type
- Order preparation time and order hour

Data cleaning, feature engineering and exploratory analysis are documented in `notebooks/notebook.ipynb`.

## Approach and results

Three models were compared on a held-out validation set: Linear Regression, Random Forest and XGBoost. The XGBoost model was then tuned and selected as the final model.

| Model | Notes |
| --- | --- |
| Linear Regression | Baseline |
| Random Forest | Better than the baseline |
| XGBoost (tuned) | **RMSE 3.77 minutes, R² 0.84** |

The tuned XGBoost model explains about 84% of the variance in delivery time, and its typical error is under four minutes. The full comparison, including the tuning steps, is in the notebook.

### Limitations

The dataset contains no information about the restaurant, the route, or real-time events such as road closures. Several features, such as traffic density and weather, are recorded as broad categories. The model therefore relies mostly on the statistical relationships present in this one dataset, and some of these relationships may not reflect real causes. It should be treated as a baseline, not as a production-ready system.

## Repository structure

```
.
├── data/
│   └── train.csv              # Training data
├── models/
│   └── model.json             # Trained XGBoost model
├── notebooks/
│   └── notebook.ipynb         # EDA, model comparison and tuning
├── train.py                   # Training script
├── predict.py                 # FastAPI prediction service
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
└── README.md
```

## How to run

### 1. Set up the environment

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Suyog-Shrestha/Food-Delivery-time-prediction.git
cd Food-Delivery-time-prediction

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Retrain the model (optional)

A trained model is already included in `models/`. To train it again from the raw data:

```bash
python train.py
```

The script reads `data/train.csv`, prepares the features, trains the XGBoost model and writes the result to the `models/` folder.

### 3. Start the prediction service

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000
```

The service is then available at `http://localhost:8000`. FastAPI also provides interactive documentation at `http://localhost:8000/docs`, where requests can be sent directly from the browser.

### 4. Send a request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

The request body must contain the input features listed in the dataset section. The exact field names and accepted values are defined in `predict.py` and shown on the `/docs` page. The response contains the predicted delivery time in minutes.

## Running with Docker

Build the image from the project root:

```bash
docker build -t food-delivery-api .
```

Run the container:

```bash
docker run -p 8000:8000 food-delivery-api
```

The service is then reachable at `http://localhost:8000`, exactly as in the local setup above.

## AI assistance disclosure

AI tools were used during this project, and their role is described here for transparency.

- An AI assistant (Claude, by Anthropic) was used to help with code, including parts of the training script, the FastAPI service and the Dockerfile, as well as debugging and the wording of this README.
- The earlier Streamlit demo in this repository was largely generated with AI assistance.
- I chose the problem and the dataset, ran the experiments, reviewed and tested the code, and checked the reported metrics myself. The modelling decisions and the conclusions are my own.

## Author

Suyog Shrestha
