import pandas as pd
import numpy as np

def calculate_elo_ratings():
    try:
        print("Loading data...")
        # Step 1: Load Data
        master = pd.read_csv("rankings_master.csv")
        current = pd.read_csv("atp_rankings_current.csv")
        
        print("Processing data...")
        # Step 2: Convert Date and handle missing values
        master['ranking_date'] = pd.to_datetime(master['ranking_date'], errors='coerce')
        master = master.dropna(subset=['ranking_date', 'player_id', 'ranking'])
        
        # Ensure ranking_points is numeric and fill missing values with 0
        master['ranking_points'] = pd.to_numeric(master['ranking_points'], errors='coerce').fillna(0)
        
        # Step 3: Sort Data by player and date
        master = master.sort_values(['player_id', 'ranking_date'])
        
        # Step 4: Initialize Elo rating system
        elo_dict = {}
        K = 32  # K-factor: determines how much ratings change
        default_elo = 1500  # Starting Elo rating for new players
        
        results = []
        
        print("Calculating Elo ratings (this may take a few minutes)...")
        # Get unique dates in chronological order
        # Ensure we have enough data points by checking the count of rankings per date
        date_counts = master['ranking_date'].value_counts().sort_index()
        # Only use dates with sufficient data (at least top 100 players)
        valid_dates = date_counts[date_counts >= 100].index
        unique_dates = sorted(valid_dates)
        
        if not unique_dates:
            raise ValueError("No valid dates with sufficient data found")
        
        for date in unique_dates:
            # Get rankings for current date
            current_rankings = master[master['ranking_date'] == date]
            
            # Sort by ranking (1 = best)
            current_rankings = current_rankings.sort_values('ranking')
            
            # Initialize Elo for new players
            for pid in current_rankings['player_id'].unique():
                if pid not in elo_dict:
                    elo_dict[pid] = default_elo
            
            # Calculate Elo changes
            for i, row in current_rankings.iterrows():
                pid = row['player_id']
                current_elo = elo_dict[pid]
                
                # Find previous ranking for this player
                prev_rank = None
                player_history = master[master['player_id'] == pid]
                prev_rankings = player_history[player_history['ranking_date'] < date]
                
                if not prev_rankings.empty:
                    # Get the most recent ranking before current date
                    prev_rank = prev_rankings.sort_values('ranking_date').iloc[-1]['ranking']
                
                # Determine performance against expected
                if prev_rank is not None:
                    # If player improved their ranking
                    if row['ranking'] < prev_rank:
                        result = 1  # Win
                    elif row['ranking'] > prev_rank:
                        result = 0  # Loss
                    else:
                        result = 0.5  # No change
                else:
                    result = 0.5  # No previous ranking
                
                # Calculate expected score against average opponent
                nearby_players = current_rankings[
                    (current_rankings['ranking'] >= row['ranking'] - 5) & 
                    (current_rankings['ranking'] <= row['ranking'] + 5) &
                    (current_rankings['player_id'] != pid)
                ]
                
                if len(nearby_players) > 0:
                    avg_opponent_elo = np.mean([elo_dict[p] for p in nearby_players['player_id']])
                    expected = 1 / (1 + 10 ** ((avg_opponent_elo - current_elo) / 400))
                else:
                    expected = 0.5
                
                # Update Elo
                elo_change = K * (result - expected)
                new_elo = current_elo + elo_change
                elo_dict[pid] = new_elo
                
                # Store result
                results.append({
                    'player_id': pid,
                    'ranking_date': date,
                    'ranking': row['ranking'],
                    'elo_rating': new_elo,
                    'elo_change': elo_change
                })
        
        # Create DataFrame from results
        elo_df = pd.DataFrame(results)
        
        # Get the latest Elo rating for each player
        latest_elo = elo_df.sort_values('ranking_date').groupby('player_id').tail(1)
        
        # Ensure player_id in current is the same type as in latest_elo
        current['player'] = current['player'].astype(int)
        latest_elo['player_id'] = latest_elo['player_id'].astype(int)
        
        # Merge with current rankings
        # First, make sure we have the latest rankings
        latest_rankings = current.sort_values('ranking_date', ascending=False).drop_duplicates('player')
        
        # Merge with Elo ratings
        merged = latest_rankings.merge(
            latest_elo[['player_id', 'elo_rating']], 
            left_on='player', 
            right_on='player_id', 
            how='left'
        )
        
        # Fill missing Elo ratings with default
        merged['elo_rating'] = merged['elo_rating'].fillna(default_elo)
        
        # Add player names from players_detail.csv if available
        try:
            players = pd.read_csv("players_detail.csv")
            merged = merged.merge(
                players[['player_id', 'name_first', 'name_last']],
                left_on='player',
                right_on='player_id',
                how='left'
            )
            # Create full name
            merged['player_name'] = merged['name_first'] + ' ' + merged['name_last']
            merged = merged.drop(['name_first', 'name_last'], axis=1)
        except Exception as e:
            print(f"Could not add player names: {str(e)}")
            merged['player_name'] = 'Unknown'
        
        # Reorder columns and add derived metrics
        merged['elo_rank'] = merged['elo_rating'].rank(ascending=False, method='min').astype(int)
        
        # Calculate performance metrics
        merged['rank_diff'] = merged['rank'] - merged['elo_rank']
        merged['performance'] = merged.apply(
            lambda x: 'Overperforming' if x['rank'] < x['elo_rank'] 
                     else 'Underperforming' if x['rank'] > x['elo_rank'] 
                     else 'On Par', 
            axis=1
        )
        
        # Final column selection and ordering
        output_cols = [
            'player', 'player_name', 'rank', 'elo_rank', 'rank_diff',
            'points', 'elo_rating', 'performance', 'ranking_date'
        ]
        output_cols = [col for col in output_cols if col in merged.columns]
        merged = merged[output_cols].sort_values('rank')
        
        # Add some summary statistics
        print("\nTop 10 Players by Elo Rating:")
        print(merged.nsmallest(10, 'elo_rank')[['rank', 'elo_rank', 'player_name', 'elo_rating']].to_string(index=False))
        
        print("\nBiggest Overperformers (Better Rank than Elo):")
        print(merged.nsmallest(5, 'rank_diff')[['rank', 'elo_rank', 'player_name', 'rank_diff']].to_string(index=False))
        
        print("\nBiggest Underperformers (Worse Rank than Elo):")
        print(merged.nlargest(5, 'rank_diff')[['rank', 'elo_rank', 'player_name', 'rank_diff']].to_string(index=False))
        
        # Save to CSV
        output_file = "atp_current_with_elo.csv"
        merged.to_csv(output_file, index=False)
        print(f"✅ Successfully created {output_file} with {len(merged)} players")
        print(f"✅ Elo ratings range from {merged['elo_rating'].min():.0f} to {merged['elo_rating'].max():.0f}")
        
        return merged
    
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        return None

# Run the function
if __name__ == "__main__":
    calculate_elo_ratings()
