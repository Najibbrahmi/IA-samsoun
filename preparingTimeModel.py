import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Step 1: Generate Synthetic Dataset
def generate_dataset():
    np.random.seed(42)
    n_rows = 2000

    data = {
        "Event_Duration_Hours": np.random.randint(1, 25, n_rows),
        "Event_Type": np.random.choice(["Hackathon", "Conference", "Workshop"], n_rows),
        "Event_Complexity": np.random.choice(["Low", "Medium", "High"], n_rows, p=[0.3, 0.5, 0.2]),
        "Number_of_Attendees": np.random.randint(10, 2001, n_rows),
        "Venue_Size": np.random.choice(["Small", "Medium", "Large"], n_rows, p=[0.4, 0.4, 0.2]),
        "Resources_Available": np.random.choice(["Low", "Medium", "High"], n_rows, p=[0.2, 0.5, 0.3]),
        "Team_Size": np.random.randint(1, 51, n_rows),
        "External_Dependencies": np.random.choice(["Yes", "No"], n_rows, p=[0.6, 0.4]),
        "Budget": np.random.choice(["Low", "Medium", "High"], n_rows, p=[0.3, 0.5, 0.2]),
        "Historical_Preparation_Time_Hours": np.random.randint(10, 500, n_rows)
    }

    df = pd.DataFrame(data)
    df["Preparation_Time_Hours"] = (
        df["Event_Duration_Hours"] * 5 +
        df["Number_of_Attendees"] * 0.1 +
        df["Team_Size"] * (-2) +
        df["Historical_Preparation_Time_Hours"] * 0.8 +
        np.random.normal(0, 10, n_rows)
    )
    df["Preparation_Time_Hours"] = df["Preparation_Time_Hours"].apply(lambda x: max(10, x)).round().astype(int)

    df.to_csv("event_preparation_dataset_2000_rows.csv", index=False)
    print("Synthetic dataset with 2000 rows generated and saved as 'event_preparation_dataset_2000_rows.csv'.")

# Step 2: Load the Dataset
def load_dataset():
    data = pd.read_csv("event_preparation_dataset_2000_rows.csv")
    print("Dataset Loaded Successfully!")
    print(data.head())
    return data

# Step 3: Preprocess the Data
def preprocess_data(data):
    categorical_features = ["Event_Type", "Event_Complexity", "Venue_Size", "Resources_Available", "External_Dependencies", "Budget"]
    numerical_features = ["Event_Duration_Hours", "Number_of_Attendees", "Team_Size", "Historical_Preparation_Time_Hours"]
    target = "Preparation_Time_Hours"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features)
        ]
    )

    X = data[categorical_features + numerical_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nPreprocessed Training Data:")
    print(X_train.head())
    print("\nPreprocessed Testing Data:")
    print(X_test.head())

    return X_train, X_test, y_train, y_test, preprocessor

# Step 4: Train the Model
def train_model(X_train, y_train, preprocessor):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    print("\nModel Trained Successfully!")
    return model

# Step 5: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

# Step 6: Save the Model
def save_model(model):
    joblib.dump(model, "event_preparation_model.pkl")
    print("\nModel Saved Successfully as 'event_preparation_model.pkl'!")

# Main Function
def main():
    # Step 1: Generate the dataset
    generate_dataset()

    # Step 2: Load the dataset
    data = load_dataset()

    # Step 3: Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)

    # Step 4: Train the model
    model = train_model(X_train, y_train, preprocessor)

    # Step 5: Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Step 6: Save the model
    save_model(model)

# Run the program
if __name__ == "__main__":
    main()