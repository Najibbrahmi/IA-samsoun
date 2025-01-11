import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Step 1: Load the dataset
data = pd.read_csv("dataset.csv")
print("Dataset Loaded Successfully!")
print(data.head())

# Step 2: Preprocess the data
categorical_features = ["Event_Type", "Event_Complexity", "Venue_Size", "Resources_Available", "External_Dependencies", "Budget"]
numerical_features = ["Event_Duration_Hours", "Number_of_Attendees", "Team_Size", "Historical_Preparation_Time_Hours"]
target = "Preparation_Time_Hours"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),  # Normalize numerical features
        ("cat", OneHotEncoder(), categorical_features)  # One-hot encode categorical features
    ]
)

# Step 3: Split the data
X = data[categorical_features + numerical_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nPreprocessed Training Data:")
print(X_train.head())
print("\nPreprocessed Testing Data:")
print(X_test.head())

# Step 4: Train the model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Step 6: Save the model
joblib.dump(model, "prep1.pkl")
print("\nModel Saved Successfully as 'prep1.pkl'!")