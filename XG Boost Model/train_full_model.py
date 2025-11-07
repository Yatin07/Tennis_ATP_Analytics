import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.steps = []
    
    def step(self, message):
        current_time = time.time()
        elapsed = current_time - self.last_checkpoint
        total_elapsed = current_time - self.start_time
        self.steps.append((message, elapsed, total_elapsed))
        print(f"[{total_elapsed:.1f}s] {message} ({elapsed:.1f}s)")
        self.last_checkpoint = current_time

def load_data(filepath, timer):
    """Load and preprocess the match data with progress tracking."""
    timer.step("Starting data loading")
    
    # Define data types for memory efficiency
    dtypes = {
        'winner_rank': 'float32', 'loser_rank': 'float32',
        'winner_age': 'float32', 'loser_age': 'float32',
        'winner_ht': 'float32', 'loser_ht': 'float32',
        'surface': 'category', 'score': 'string',
        'tourney_date': 'string', 'year': 'int32'
    }
    
    # Read the CSV file with only necessary columns
    usecols = [
        'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
        'winner_ht', 'loser_ht', 'surface', 'score', 'tourney_date', 'year',
        'winner_name', 'loser_name'  # Added for consistency with Power BI
    ]
    
    # Read data in chunks
    chunk_size = 100000
    chunks = []
    
    # Get total rows for progress bar
    total_rows = sum(1 for _ in open(filepath, 'r', encoding='utf-8')) - 1  # subtract header
    
    # Read in chunks
    with tqdm(total=total_rows, desc="Processing matches") as pbar:
        for chunk in pd.read_csv(
            filepath, 
            usecols=usecols,
            dtype=dtypes, 
            chunksize=chunk_size,
            low_memory=False
        ):
            # Basic cleaning
            chunk = chunk.dropna(subset=['winner_rank', 'loser_rank', 'surface'])
            
            # Remove walkovers/retirements
            if 'score' in chunk.columns:
                chunk = chunk[~chunk['score'].str.contains('RET|W/O|DEF', na=False)]
            
            chunks.append(chunk)
            pbar.update(chunk_size)  # Approximate progress
    
    # Combine chunks
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates()
        timer.step(f"Processed {len(df):,} matches after cleaning")
        return df
    else:
        raise ValueError("No valid data found in the input file")

def create_features(df):
    """Create features for the model."""
    X = pd.DataFrame()
    
    # Basic features - matching Power BI visualization
    # Note: Using winner_rank - loser_rank to match Power BI's calculation
    X['rank_diff'] = df['winner_rank'] - df['loser_rank']
    
    # Handle missing values for age and height
    if 'winner_age' in df.columns and 'loser_age' in df.columns:
        X['age_diff'] = df['winner_age'] - df['loser_age'].fillna(df['winner_age'])
    
    # Match column names with Power BI (winner_ht -> winner_height)
    if 'winner_ht' in df.columns and 'loser_ht' in df.columns:
        X['ht_diff'] = df['winner_ht'] - df['loser_ht'].fillna(df['winner_ht'])
        
    # Add year if available (for recency weighting)
    if 'year' in df.columns:
        X['year'] = df['year']
    
    # Encode surface
    le = LabelEncoder()
    X['surface_encoded'] = le.fit_transform(df['surface'].fillna('Hard'))
    
    # Add year as a feature if available
    if 'year' in df.columns:
        X['year'] = df['year']
    
    # Create balanced target variable (1 for winner, 0 for loser)
    y = np.ones(len(df))  # All 1s for original winners
    
    # Create swapped version for balanced classes
    X_swapped = X.copy()
    X_swapped['rank_diff'] = -X['rank_diff']
    
    if 'age_diff' in X.columns:
        X_swapped['age_diff'] = -X['age_diff']
    if 'ht_diff' in X.columns:
        X_swapped['ht_diff'] = -X['ht_diff']
    
    # Combine original and swapped data
    X_balanced = pd.concat([X, X_swapped])
    y_balanced = np.concatenate([y, np.zeros(len(y))])
    
    # Shuffle the data
    idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced.iloc[idx].reset_index(drop=True)
    y_balanced = y_balanced[idx]
    
    return X_balanced, y_balanced, le

def train_model(X, y, timer):
    """Train the XGBoost model with cross-validation."""
    timer.step("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set parameters for faster training with good performance
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': ['logloss', 'error'],
        'n_jobs': -1
    }
    
    # Cross-validation
    print("\n=== Cross-Validation ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(**params), 
        X, y, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1
    )
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    timer.step("Training final model")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    return model, (X_test, y_test)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Player 2 Wins', 'Player 1 Wins']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                {'Player 2':<12} {'Player 1':<12}")
    print(f"Actual Player 2 {cm[0,0]:<12} {cm[0,1]:<12}")
    print(f"       Player 1 {cm[1,0]:<12} {cm[1,1]:<12}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = model.get_score(importance_type='weight')
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.2f}")
    
    return {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'feature_importance': {k: float(v) for k, v in feature_importance.items()}
    }

def save_model(model, le, metrics, output_dir='.'):
    """Save the model and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in JSON format
    model_path = os.path.join(output_dir, 'tennis_model_full.json')
    
    # Get the actual features used in the model
    feature_names = ['rank_diff', 'age_diff', 'ht_diff', 'surface_encoded', 'year']
    # Only keep features that exist in the model
    existing_features = [f for f in feature_names if f in model.feature_names]
    model.feature_names = existing_features
    
    model.save_model(model_path)
    
    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'surface_encoder': le.classes_.tolist(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_info': 'Trained to match Power BI visualization requirements',
        'feature_descriptions': {
            'rank_diff': 'winner_rank - loser_rank',
            'age_diff': 'winner_age - loser_age',
            'ht_diff': 'winner_ht - loser_ht',
            'surface_encoded': 'Encoded surface type (0=Hard, 1=Clay, 2=Grass, etc.)'
        }
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    return {
        'model_path': os.path.abspath(model_path),
        'metadata_path': os.path.abspath(metadata_path),
        'feature_importance_plot': os.path.abspath(os.path.join(output_dir, 'feature_importance.png'))
    }

def main():
    # Initialize timer
    timer = Timer()
    
    try:
        # Configuration
        DATA_FILE = 'matches_master.csv'
        OUTPUT_DIR = 'model_output'
        
        print(f"{'='*50}")
        print(f"  Tennis Match Prediction Model - Full Dataset")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")
        
        # Load and preprocess data
        df = load_data(DATA_FILE, timer)
        
        # Create features
        timer.step("Creating features")
        X, y, le = create_features(df)
        print(f"\nCreated balanced dataset with {len(X):,} samples")
        
        # Create a balanced dataset
        print(f"Using all {len(X):,} matches for training")
        
        # Train model
        model, (X_test, y_test) = train_model(X, y, timer)
        
        # Evaluate model
        timer.step("Evaluating model")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model and metadata
        timer.step("Saving model and metadata")
        saved_files = save_model(model, le, metrics, OUTPUT_DIR)
        
        # Print summary
        print("\n" + "="*50)
        print("  Model Training Complete!")
        print(f"  Total time: {time.time() - timer.start_time:.1f} seconds")
        print(f"  Model saved to: {saved_files['model_path']}")
        print(f"  Metadata saved to: {saved_files['metadata_path']}")
        print(f"  Feature importance plot: {saved_files['feature_importance_plot']}")
        print(f"  Final Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"{'='*50}")
        
        # Save timing information
        timing_info = [
            {
                'step': step,
                'step_time': float(step_time),
                'total_time': float(total_time)
            }
            for step, step_time, total_time in timer.steps
        ]
        
        with open(os.path.join(OUTPUT_DIR, 'training_timing.json'), 'w') as f:
            json.dump(timing_info, f, indent=2)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
