import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os
import time
from datetime import datetime, timedelta
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "tennis_model_full.xgb"
SAMPLE_SIZE = None  # Set to None to use full dataset
RANDOM_STATE = 42
CHUNK_SIZE = 50000  # Process data in chunks to handle memory
np.random.seed(RANDOM_STATE)

class Timer:
    def __init__(self, total_steps=10):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self, message=""):
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        avg_time_per_step = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        eta = avg_time_per_step * remaining_steps
        
        print(f"\n[Step {self.current_step}/{self.total_steps}] {message}")
        print(f"  ↳ Elapsed: {timedelta(seconds=int(elapsed))} | "
              f"ETA: {timedelta(seconds=int(eta))} | "
              f"Avg: {avg_time_per_step:.1f}s/step")
        
        self.last_time = current_time
        return elapsed

def load_and_prepare_data(filepath, timer, sample_size=None):
    """Load and preprocess the match data with progress tracking."""
    timer.step("Starting data loading")
    
    # Define data types for memory efficiency
    dtypes = {
        'winner_id': 'int32', 'loser_id': 'int32',
        'winner_rank': 'float32', 'loser_rank': 'float32',
        'winner_age': 'float32', 'loser_age': 'float32',
        'winner_ht': 'float32', 'loser_ht': 'float32',
        'surface': 'category', 'score': 'string',
        'tourney_date': 'string', 'tourney_name': 'category'
    }
    
    # Get total file size for progress tracking
    total_size = os.path.getsize(filepath)
    
    # Read in chunks for memory efficiency
    timer.step("Reading and processing data in chunks")
    chunks = []
    processed_rows = 0
    
    # First pass: determine total rows for progress tracking
    with tqdm(total=total_size, unit='B', unit_scale=True, 
              desc="Processing data") as pbar:
        for chunk in pd.read_csv(filepath, dtype=dtypes, chunksize=CHUNK_SIZE):
            # Basic cleaning
            chunk = chunk.dropna(subset=['winner_rank', 'loser_rank', 'surface'])
            
            # Remove walkovers/retirements
            chunk = chunk[~chunk['score'].str.contains('RET|W/O|DEF', na=False)]
            
            # Remove duplicate matches
            chunk = chunk.drop_duplicates()
            
            chunks.append(chunk)
            processed_rows += len(chunk)
            pbar.update(CHUNK_SIZE * 1000)  # Approximate bytes per row
            
            # Early stop if sampling
            if sample_size and processed_rows >= sample_size * 1.5:  # Buffer for filtering
                break
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=RANDOM_STATE)
    
    timer.step(f"Processed {len(df):,} matches after cleaning")
    return df
    
    # Data cleaning
    timer.step("Cleaning data")
    critical_cols = ['winner_id', 'loser_id', 'winner_rank', 'loser_rank', 
                    'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'surface']
    df = df.dropna(subset=critical_cols)
    df = df[~df['score'].str.contains('RET|W/O|DEF', na=False, case=False)]
    
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        timer.step(f"Sampled to {SAMPLE_SIZE:,} matches")
    
    # Feature engineering
    timer.step("Creating features")
    df['rank_diff'] = df['loser_rank'] - df['winner_rank']
    df['age_diff'] = df['winner_age'] - df['loser_age']
    df['ht_diff'] = df['winner_ht'] - df['loser_ht']
    
    le = LabelEncoder()
    df['surface_encoded'] = le.fit_transform(df['surface'].astype(str))
    df['target'] = (df['rank_diff'] > 0).astype(int)
    
    X = df[['rank_diff', 'age_diff', 'ht_diff', 'surface_encoded']]
    y = df['target']
    
    metadata = {
        'feature_names': X.columns.tolist(),
        'surface_encoder': le,
        'sample_size': len(df),
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return X, y, metadata

def evaluate_model(model, X_test, y_test):
    """Evaluate model with comprehensive metrics."""
    # Get predictions and probabilities
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_proba = model.predict(X_test)
    
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
    for i, score in enumerate(model.get_score(importance_type='weight').items()):
        print(f"{score[0]}: {score[1]:.2f}")
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def train_model(X, y, timer):
    """Train the model with progress tracking and cross-validation."""
    timer.step("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Configure callbacks for progress tracking
    timer.step("Training model (this may take a few minutes)")
    
    # Use DMatrix for better memory efficiency
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'eval_metric': ['logloss', 'error'],
        'n_jobs': -1  # Use all available cores
    }
    
    # Skip cross-validation for faster training on full dataset
    print("\nSkipping cross-validation for faster training on full dataset")
    print("Using train-test split for evaluation instead")
    
    # Train final model with progress tracking
    evals_result = {}
    
    # Estimate training time (roughly 1 second per 1000 rows)
    est_train_seconds = max(30, len(X_train) // 1000)
    print(f"\n{'='*50}")
    print(f"  Starting model training")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_test):,}")
    print(f"  Features: {X_train.columns.tolist()}")
    print(f"  Estimated training time: {est_train_seconds} seconds")
    print(f"  Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train with progress tracking
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,  # Increased for better convergence
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=10,  # Show progress every 10 iterations
        callbacks=[
            xgb.callback.EarlyStopping(rounds=20, save_best=True)
        ]
    )
    
    train_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"  Training completed in {train_time:.1f} seconds")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")
    print(f"  End time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}\n")
    
    # Evaluate model
    evaluation = evaluate_model(model, dtest, y_test)
    
    return model, (X_test, y_test, evaluation)

def main():
    print("=== Tennis Match Prediction Model Training ===\n")
    
    # Initialize timer with estimated steps
    timer = Timer(total_steps=8)  # Adjust based on actual steps
    
    try:
        # 1. Load and prepare data
        data_path = "matches_master.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {os.path.abspath(data_path)}")
        
        # Get file size for time estimation
        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        est_time_min = max(1, int(file_size_mb / 50))  # Rough estimate: 50MB/min
        print(f"\n{'='*50}\n")
        print(f"  Processing full dataset (~{file_size_mb:.1f} MB)")
        print(f"  Estimated processing time: {est_time_min} minutes")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{'='*50}\n")
        
        # Load and prepare data
        df = load_and_prepare_data(data_path, timer, sample_size=SAMPLE_SIZE)
        
        # Create features
        timer.step("Creating features")
        X = pd.DataFrame()
        X['rank_diff'] = df['loser_rank'] - df['winner_rank']
        X['age_diff'] = df['winner_age'] - df['loser_age']
        X['ht_diff'] = df['winner_ht'] - df['loser_ht']
        
        # Encode surface
        le = LabelEncoder()
        X['surface_encoded'] = le.fit_transform(df['surface'].fillna('Hard'))
        
        # Create balanced target variable (1 for winner, 0 for loser)
        # We'll use 1 for the original winner and 0 for the original loser
        y = np.ones(len(df))  # All 1s for original winners
        
        # For binary classification, we need both classes (0 and 1)
        # So we'll duplicate the data with swapped players for the other class
        X_swapped = X.copy()
        X_swapped['rank_diff'] = -X['rank_diff']
        X_swapped['age_diff'] = -X['age_diff']
        X_swapped['ht_diff'] = -X['ht_diff']
        
        # Combine original and swapped data
        X_balanced = pd.concat([X, X_swapped])
        y_balanced = np.concatenate([y, np.zeros(len(y))])
        
        # Shuffle the data
        idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced.iloc[idx].reset_index(drop=True)
        y_balanced = y_balanced[idx]
        
        print(f"Created balanced dataset with {len(X_balanced):,} samples")
        
        metadata = {
            'feature_names': X.columns.tolist(),
            'surface_encoder': le,
            'sample_size': len(df),
            'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features_used': ['rank_diff', 'age_diff', 'ht_diff', 'surface_encoded']
        }
        
        # 2. Train model on balanced data
        model, (X_test, y_test, evaluation) = train_model(X_balanced, y_balanced, timer)
        
        # 3. Save the model and metadata
        timer.step("Saving model")
        # Save in XGBoost binary format
        model.save_model(MODEL_PATH)
        # Save in JSON format
        model.save_model('tennis_model.json')
        joblib.dump(metadata, 'model_metadata.pkl')
        
        # 4. Example prediction
        timer.step("Testing with example prediction")
        example = {
            'player1_rank': 5, 'player2_rank': 10,
            'player1_age': 25, 'player2_age': 23,
            'player1_ht': 188, 'player2_ht': 185,
            'surface': 'Hard'
        }
        
        # Make prediction
        X_example = np.array([[
            example['player2_rank'] - example['player1_rank'],  # rank_diff
            example['player1_age'] - example['player2_age'],    # age_diff
            example['player1_ht'] - example['player2_ht'],      # ht_diff
            metadata['surface_encoder'].transform([example['surface']])[0]  # surface
        ]])
        
        # Make prediction with feature names
        dtest = xgb.DMatrix(X_example, feature_names=['rank_diff', 'age_diff', 'ht_diff', 'surface_encoded'])
        prob = model.predict(dtest)[0]
        
        print("\n=== Example Prediction ===")
        print(f"Player 1 (Rank {example['player1_rank']}) vs "
              f"Player 2 (Rank {example['player2_rank']}) on {example['surface']}")
        print(f"Player 1 Win Probability: {prob:.2%}")
        
        # 5. Save feature importance plot
        timer.step("Generating feature importance plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(model, ax=ax)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nFeature importance plot saved as 'feature_importance.png'")
        plt.close(fig)
        
        # Final timing
        total_time = time.time() - timer.start_time
        print(f"\n=== Training completed in {timedelta(seconds=int(total_time))} ===")
        print(f"Model saved to: {os.path.abspath(MODEL_PATH)}")
        print(f"Model also saved in JSON format to: {os.path.abspath('tennis_model.json')}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
