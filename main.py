import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('Battery_RUL.csv')
df.head()

# Check dataset information
print(df.info())

# Check for missing values
if df.isnull().sum().sum() > 0:
    print("Dataset contains missing values. Removing...")
    df = df.dropna()
else:
    print('No Missing Vaalues')

# Define features (X) and target (y)
X = df.drop(columns=["RUL"], 'Unnamed: 0'])
y = df["RUL"]

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
columns_to_scale = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 
                    'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']

X_scaled_df = pd.DataFrame(X_scaled, columns=columns_to_scale)
X_scaled_df.head()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Models

# Initialize Linear Regression model
linear_regressor = LinearRegression()

# Train Linear Regression model
linear_regressor.fit(X_train, y_train)

# Predict using Linear Regression
y_pred_linear = linear_regressor.predict(X_test)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Evaluate Linear Regression model
linear_metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

results_lr = []
results_lr.append(linear_metrics)

results_lr_df = pd.DataFrame(results_lr)

# Display results in a simple tabular format
print("\nLinear Regressor Metrics:")
print(results_lr_df)


# Create a pipeline for Polynomial Regression with Ridge Regularization
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))

# Train the model
model.fit(X_train, y_train)

# Evaluate Polynomial Regressor
poly_metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

results_pr = []
results_pr.append(poly_metrics)

results_pr_df = pd.DataFrame(results_pr)

# Display results in a simple tabular format
print("\nPolynomial Regressor Metrics(n = 2):")
print(results_pr_df)


# Polynomial Regression : n = 3
model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), Ridge(alpha=1.0))

# Train the model
model.fit(X_train, y_train)

# Evaluate Polynomial Regressor
poly_metrics_3 = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

results_pr_3 = []
results_pr_3.append(poly_metrics_3)

results_pr_3_df = pd.DataFrame(results_pr_3)

# Display results in a simple tabular format
print("\nPolynomial Regressor Metrics(n = 3):")
print(results_pr_3_df)

# Polynomial Regression : n = 4

model = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), Ridge(alpha=1.0))

# Train the model
model.fit(X_train, y_train)

# Evaluate Decision Tree Regressor
poly_metrics_4 = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

results_pr_4 = []
results_pr_4.append(poly_metrics_4)

results_pr_4_df = pd.DataFrame(results_pr_4)

# Display results in a simple tabular format
print("\nPolynomial Regressor Metrics(n = 4):")
print(results_pr_4_df)

# Initialize Decision Tree Regressor
decision_tree_regressor = DecisionTreeRegressor(random_state=42)

# Train Decision Tree Regressor
decision_tree_regressor.fit(X_train, y_train)

# Predict using Decision Tree Regressor
y_pred_tree = decision_tree_regressor.predict(X_test)

# Evaluate Decision Tree Regressor
tree_metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}

results_dt = []
results_dt.append(tree_metrics)

results_dt_df = pd.DataFrame(results_dt)

# Display results in a simple tabular format
print("\nDecision Tree Regressor Metrics:")
print(results_dt_df)

# Initialize the Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model on the training set
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluate the Random Forest model
rf_metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2  # R-Squared
}

results_rf = []
results_rf.append(rf_metrics)

results_rf_df = pd.DataFrame(results_rf)

# Display results in a simple tabular format
print("\nRandom Forest Metrics:")
print(results_rf_df)


# Random Forest - Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Create all combinations of parameters
grid = ParameterGrid(param_grid)

# Initialize a list to store results
results_rf_2 = []

# Perform grid search manually
for params in grid:
    # Update the model with the current set of parameters
    rf.set_params(**params)
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Store results
    results_rf_2.append({
        'Parameters': params,
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2
    })

# Convert results to a sorted DataFrame
results_rf_2_df = pd.DataFrame(results_rf_2)
results_rf_2_df = results_rf_2_df.sort_values(by='MAE', ascending=True)

# Display results in a simple tabular format
print("\nRandom Forest Metrics (Hyperparameter Tuned):")
print(results_rf_2_df)

# Gradient Boosting

# Define the Gradient Boosting Regressor
gb = GradientBoostingRegressor(
    n_estimators=200,    # Number of trees
    learning_rate=0.1,   # Learning rate
    max_depth=3,         # Maximum depth of trees
    subsample=0.8,       # Fraction of samples used for fitting individual trees
    random_state=42      # For reproducibility
)

# Train the model
gb.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb.predict(X_test)

# Evaluate the model
gb_metrics = {
    "MAE": mse,
    "RMSE": rmse,
    "R2": r2
}
results_gb = []
results_gb.append(gb_metrics)
 # Convert results to a sorted DataFrame
results_gb_df = pd.DataFrame(results_gb)
results_gb_df = results_gb_2_df.sort_values(by='MAE', ascending=True)

# Display results in a simple tabular format
print("\Gradient Boosting metrics:")
print(results_gb_df)

# Gradient Boosting -Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Initialize the Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)

# Perform Grid Search
grid_search_gb = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1,
    scoring='neg_mean_absolute_error'  # Metric for optimization
)

# Fit GridSearchCV
grid_search_gb.fit(X_train, y_train)

# Get the best parameters and model
best_params_gb = grid_search_gb.best_params_
best_gb_model = grid_search_gb.best_estimator_

# Evaluate the tuned Gradient Boosting model
y_pred_tuned_gb = best_gb_model.predict(X_test)
tuned_gb_metrics = {
    "MAE": mse,
    "RMSE": rmse,
    "R2": r2
}
results_gb_2 = []
results_gb_2.append(tuned_gb_metrics)
 # Convert results to a sorted DataFrame
results_gb_2_df = pd.DataFrame(results_gb_2)
results_gb_2_df = results_gb_2_df.sort_values(by='MAE', ascending=True)

# Display results in a simple tabular format
print("\Tuned Gradient Boosting metrics:")
print(results_gb_2_df)

