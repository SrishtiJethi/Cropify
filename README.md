### Crop Recommendation System - Cropify

This project aims to develop a crop recommendation system using an ensemble of machine learning models. The system predicts the most suitable crop to grow based on various environmental factors such as nitrogen (N), phosphorus (P), potassium (K) levels, temperature, humidity, pH, and rainfall. The dataset used for this project is the "Crop_recommendation.csv".

#### Project Overview

1. **Data Preprocessing**:
    - The dataset is loaded and a dictionary is created to map crop names to numerical labels.
    - Features (X) and the target variable (y) are extracted, and the data is split into training and testing sets.
    - Features are normalized using `StandardScaler`.

2. **Model Training**:
    - Two base models are trained: a `RandomForestClassifier` and a `LogisticRegression` model, both with regularization to prevent overfitting.
    - The predictions from these base models on the test set are combined to form a new feature matrix.

3. **Meta-Model Training**:
    - A meta-model (`LogisticRegression`) is trained on the combined predictions of the base models.
    - The meta-model helps improve the prediction accuracy by leveraging the strengths of both base models.

4. **Prediction Function**:
    - The `recommendation_ensemble` function takes input parameters (N, P, K, temperature, humidity, pH, and rainfall) and predicts the most suitable crop.
    - The function scales the input features, makes predictions using the base models, combines these predictions, and then uses the meta-model to make the final crop recommendation.

#### Project Dependencies

- `numpy`
- `pandas`
- `scikit-learn`

#### How to Run

1. Clone the repository.
2. Ensure the dependencies are installed: `pip install numpy pandas scikit-learn`.
3. Place the `Crop_recommendation.csv` file in the project directory.
4. Run the script to see crop recommendations based on input environmental factors.

This ensemble-based approach leverages the strengths of multiple models to provide accurate and reliable crop recommendations, aiding farmers and agricultural planners in making informed decisions.
