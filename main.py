import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("Battery_RUL.csv")

# Check dataset information
print(data.info())

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Dataset contains missing values. Removing...")
    data = data.dropna()

# Define features (X) and target (y)
X = data.drop(columns=["RUL"])
y = data["RUL"]

# Normalize features using MinMaxScaler
print("Normalizing features using MinMaxScaler...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models dictionary for looping
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "kNN": KNeighborsRegressor(n_neighbors=5),
    "SVM": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
}

# Function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\nTraining and Evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} Performance:")
    print(f"  MAE: {mae}")
    print(f"  RMSE: {rmse}")
    print(f"  R2 Score: {r2}")
    
    return y_pred, {"MAE": mae, "RMSE": rmse, "R2 Score": r2}

# Store results
results = {}
predictions = {}

# Train and evaluate each model
for name, model in models.items():
    y_pred, metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results[name] = metrics
    predictions[name] = y_pred

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results).T
print("\nModel Performance Summary:")
print(results_df)

# Save results to a CSV file
results_df.to_csv("model_performance.csv", index=True)

# Visualizations
print("\nGenerating Visualizations...")

# RUL Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data["RUL"], kde=True, bins=30)
plt.title("RUL Distribution")
plt.xlabel("RUL")
plt.ylabel("Frequency")
plt.savefig("rul_distribution.png")
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_correlation.png")
plt.show()

# Actual vs Predicted for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions["Random Forest"], alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.title("Actual vs Predicted RUL (Random Forest)")
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.savefig("actual_vs_predicted_rf.png")
plt.show()

# Interactive Visualizations (Plotly)
fig = px.scatter(x=y_test, y=predictions["Random Forest"], labels={"x": "Actual RUL", "y": "Predicted RUL"}, title="Interactive Scatter Plot (Random Forest)")
fig.write_html("interactive_scatter_rf.html")
print("Visualizations saved!")

print("\nAll tasks completed successfully!")
