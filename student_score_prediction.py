
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create a small dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 40, 50, 55, 60, 65, 70, 75, 85, 95]
}

df = pd.DataFrame(data)

# Step 2: Split data into training and testing sets
X = df[['Hours_Studied']]
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print("📊 Mean Absolute Error:", mae)

# Step 6: Test custom prediction
hours = 7.5
predicted_score = model.predict([[hours]])
print(f"📈 Predicted score for {hours} study hours = {predicted_score[0]:.2f}")

# Optional: Save model for reuse
import joblib
joblib.dump(model, "student_score_model.pkl")
print("✅ Model saved as student_score_model.pkl")
