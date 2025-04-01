from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src.data_preprocess import preprocess_data

def train_model(X_train, y_train, model_file):
    # Train Random Forest model for regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = rf_model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
  
    
    # Save the trained model
    joblib.dump(rf_model, model_file)

if __name__ == "__main__":
    # Define paths
    data_file = 'dataset.json'
    model_file = 'trained_model.pkl'

    # Preprocess data
    X, y = preprocess_data(data_file)

    # Split data into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    train_model(X_train, y_train, model_file)