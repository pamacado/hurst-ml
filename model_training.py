import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib # Used to save the trained model to a file

# We import the functions from our previous module!
from data_pipeline import fetch_data, calculate_all_features

def perform_walk_forward_validation(X, y, model, n_splits=5):
    """
    Simulates real-world trading by training on the past and testing on the future.
    Prevents Data Leakage using TimeSeriesSplit.
    """
    # TimeSeriesSplit creates expanding windows.
    # E.g., Fold 1: Train [2005-2010], Test [2011]
    #       Fold 2: Train [2005-2011], Test [2012] ...
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_accuracies = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        # Split the data logically (no random shuffling!)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model on the past
        model.fit(X_train, y_train)
        
        # Predict the future
        predictions = model.predict(X_test)
        
        # Grade the exam
        accuracy = accuracy_score(y_test, predictions)
        fold_accuracies.append(accuracy)
        
        print(f"Fold {fold + 1} | Train Size: {len(X_train)} | Test Size: {len(X_test)} | Accuracy: {accuracy:.4f}")
        
    avg_accuracy = np.mean(fold_accuracies)
    print(f">>> Average Walk-Forward Accuracy: {avg_accuracy:.4f}")
    
    return avg_accuracy

def get_feature_importance(model, feature_names):
    """
    Extracts the Gini Impurity reduction for each feature to see what the model actually learned.
    """
    importances = model.feature_importances_
    
    # Create a DataFrame to sort and display cleanly
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- FEATURE IMPORTANCE ---")
    print(importance_df.to_string(index=False))
    return importance_df

if __name__ == "__main__":
    print("1. Loading and preparing data from Phase 1 (2005 to 2024)...")
    raw_df = fetch_data(ticker="SPY", start_date="2005-01-01", end_date="2025-01-01")
    df = calculate_all_features(raw_df)
    
    feature_columns = ['Log_Ret_1d', 'Vol_20d', 'Mom_10d', 'Rel_Vol', 'Hurst_60d']
    X = df[feature_columns]
    y = df['Target']
    
    print("\n2. Initializing the Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    print("\n3. Running Walk-Forward Optimization (Full Model)...")
    full_accuracy = perform_walk_forward_validation(X, y, rf_model, n_splits=5)
    
    print("\n4. Auditing the Model (Feature Importance)...")
    rf_model.fit(X, y)
    get_feature_importance(rf_model, feature_columns)
    
    # --- ABLATION STUDIES ---
    print("\n5a. Running Ablation Study A (Removing Hurst_60d)...")
    X_no_hurst = X[['Log_Ret_1d', 'Vol_20d', 'Mom_10d', 'Rel_Vol']]
    acc_no_hurst = perform_walk_forward_validation(X_no_hurst, y, rf_model, n_splits=5)
    diff_hurst = full_accuracy - acc_no_hurst
    print(f">>> Impact of Hurst on Accuracy: {diff_hurst:+.4f} ({diff_hurst*100:+.2f}%)")
    
    print("\n5b. Running Ablation Study B (Removing Mom_10d, keeping Hurst)...")
    # We remove Momentum to see if Hurst can carry the trend-following weight alone
    X_no_mom = X[['Log_Ret_1d', 'Vol_20d', 'Rel_Vol', 'Hurst_60d']]
    acc_no_mom = perform_walk_forward_validation(X_no_mom, y, rf_model, n_splits=5)
    diff_mom = full_accuracy - acc_no_mom
    print(f">>> Impact of Momentum on Accuracy: {diff_mom:+.4f} ({diff_mom*100:+.2f}%)")
    # ------------------------
    
    print("\n6. Re-training the final model on ALL features before saving...")
    rf_model.fit(X, y) 
    
    joblib.dump(rf_model, 'quant_random_forest.pkl')
    print("Model saved as 'quant_random_forest.pkl'. Ready for Backtesting!")