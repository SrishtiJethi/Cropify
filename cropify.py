import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Load the data
crop = pd.read_csv("Crop_recommendation.csv")

# Create a dictionary to map crop names to numerical labels
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

crop['crop_num'] = crop['label'].map(crop_dict)

# Features (X) and target variable (y)
X = crop.drop(['crop_num', 'label'], axis=1)
y = crop['crop_num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model with regularization
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rfc.fit(X_train_scaled, y_train)

# Train a Logistic Regression model with regularization
lr = LogisticRegression(C=0.1)  # C is the regularization parameter
lr.fit(X_train_scaled, y_train)

# Make predictions on the test set
rfc_pred = rfc.predict(X_test_scaled)
lr_pred = lr.predict(X_test_scaled)

# Create a new feature matrix with predictions from both models
X_test_ensemble = np.column_stack((rfc_pred, lr_pred))

# Train a meta-model (Logistic Regression) on the combined predictions
meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=6000)
meta_model.fit(X_test_ensemble, y_test)

def recommendation_ensemble(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = scaler.transform(features)
    rfc_prediction = rfc.predict(transformed_features)
    lr_prediction = lr.predict(transformed_features)
    ensemble_features = np.column_stack((rfc_prediction, lr_prediction))
    ensemble_prediction = meta_model.predict(ensemble_features).reshape(1, -1)

    # Take the most common prediction among the ensemble models
    final_prediction = np.argmax(np.bincount(ensemble_prediction[0]))

    return final_prediction

# Example usage
N = 35
P = 23
k = 43
temperature = 30.0
humidity = 57
ph = 3
rainfall = 100

ensemble_predict = recommendation_ensemble(N, P, k, temperature, humidity, ph, rainfall)

# Mapping back to crop names
if ensemble_predict in crop_dict.values():
    recommended_crop = [key for key, value in crop_dict.items() if value == ensemble_predict][0]
    print("{} is a recommended crop based on the ensemble model.".format(recommended_crop))
else:
    print("Sorry, we are not able to recommend a proper crop for this environment.")
