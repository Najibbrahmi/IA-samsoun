import pandas as pd
import joblib

# Function to convert hours into months, days, and hours
def convert_hours_to_readable(hours):
    """
    Convert hours into months, days, and hours.
    Assumptions:
    - 1 month = 30 days
    - 1 day = 24 hours
    """
    months = hours // (30 * 24)  # Calculate months
    remaining_hours = hours % (30 * 24)  # Remaining hours after calculating months
    days = remaining_hours // 24  # Calculate days
    hours_remaining = remaining_hours % 24  # Remaining hours after calculating days

    return f"{int(months)} months, {int(days)} days, {int(hours_remaining)} hours"

# Step 1: Load the saved model
model = joblib.load("prep1.pkl")
print("Model Loaded Successfully!")

# Step 2: Prepare new input data
# Example new data (replace with your own data)
new_data = pd.DataFrame({
    "Event_Duration_Hours": [10],  # Example: 10 hours
    "Event_Type": ["Hackathon"],  # Example: Hackathon
    "Event_Complexity": ["High"],  # Example: High complexity
    "Number_of_Attendees": [500],  # Example: 500 attendees
    "Venue_Size": ["Large"],  # Example: Large venue
    "Resources_Available": ["High"],  # Example: High resources
    "Team_Size": [20],  # Example: 20 team members
    "External_Dependencies": ["Yes"],  # Example: External dependencies exist
    "Budget": ["High"],  # Example: High budget
    "Historical_Preparation_Time_Hours": [300]  # Example: 300 hours historical prep time
})

# Step 3: Make predictions
predicted_time = model.predict(new_data)
print(f"Predicted Preparation Time (in hours): {predicted_time[0]} hours")

# Step 4: Convert predicted time to readable format
readable_time = convert_hours_to_readable(predicted_time[0])
print(f"Predicted Preparation Time (readable): {readable_time}")