import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def load_data(filepath):
    # Read only necessary columns
    cols = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age',
            'winner_ht', 'loser_ht', 'surface', 'score', 'tourney_date']
    
    # Read data in chunks for memory efficiency
    chunks = []
    for chunk in pd.read_csv(filepath, usecols=cols, chunksize=100000, low_memory=False):
        # Basic cleaning
        chunk = chunk.dropna(subset=['winner_rank', 'loser_rank', 'surface'])
        chunk = chunk[~chunk['score'].str.contains('RET|W/O|DEF', na=False)]
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    return df

def create_features(df):
    # Create features in the exact order needed for prediction
    X = pd.DataFrame()
    
    # Match the exact calculation from Power BI
    X['rank_diff'] = df['loser_rank'] - df['winner_rank']
    X['age_diff'] = df['winner_age'] - df['loser_age']
    X['ht_diff'] = df['winner_ht'] - df['loser_ht']
    
    # Encode surface (Hard=0, Clay=1, Grass=2, etc.)
    le = LabelEncoder()
    X['surface_encoded'] = le.fit_transform(df['surface'].fillna('Hard'))
    
    # Create target (1 if first player wins, 0 if second player wins)
    y = np.ones(len(df))
    
    # Create balanced dataset by adding reversed matches
    X_swapped = X.copy()
    X_swapped['rank_diff'] = -X['rank_diff']
    X_swapped['age_diff'] = -X['age_diff']
    X_swapped['ht_diff'] = -X['ht_diff']
    
    X_balanced = pd.concat([X, X_swapped])
    y_balanced = np.concatenate([y, 1-y])
    
    return X_balanced, y_balanced, le

def train_and_save_model():
    # Load and prepare data
    print("Loading data...")
    df = load_data('matches_master.csv')
    
    print("Creating features...")
    X, y, le = create_features(df)
    
    # Ensure features are in the correct order
    feature_order = ['rank_diff', 'age_diff', 'ht_diff', 'surface_encoded']
    X = X[feature_order]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create DMatrix with feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_order)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_order)
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': ['logloss', 'error']
    }
    
    # Train model
    print("Training model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Ensure feature names are set
    model.feature_names = feature_order
    
    # Save model
    model.save_model('tennis_model.json')
    print("\nModel saved as 'tennis_model.json'")
    
    # Print surface encoding for reference
    print("\nSurface encoding:")
    for code, surface in enumerate(le.classes_):
        print(f"{surface}: {code}")
    
    return model

if __name__ == "__main__":
    train_and_save_model()
