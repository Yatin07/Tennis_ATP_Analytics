import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

def calculate_elo_ratings():
    try:
        print("🚀 Starting Elo rating calculation...")
        start_time = datetime.now()
        
        print("📊 Loading data...")
        # Load data with optimized dtypes
        dtypes = {
            'ranking_date': 'str',
            'player_id': 'int32',
            'ranking': 'int16',
            'ranking_points': 'float32'
        }
        usecols = ['ranking_date', 'player_id', 'ranking', 'ranking_points']
        
        # Read in chunks for memory efficiency
        chunks = pd.read_csv("rankings_master.csv", 
                            usecols=usecols,
                            dtype=dtypes,
                            chunksize=100000)
        
        # Process chunks and combine
        master = pd.concat(chunks, ignore_index=True)
        
        # Load current rankings
        current = pd.read_csv("atp_rankings_current.csv", 
                             dtype={'player': 'int32', 'rank': 'int16', 'points': 'int32'})
        
        print("⚙️  Processing data...")
        # Convert date once
        master['ranking_date'] = pd.to_datetime(master['ranking_date'], errors='coerce')
        master = master.dropna(subset=['ranking_date', 'player_id', 'ranking'])
        
        # Sort by date and player for faster lookups
        master = master.sort_values(['player_id', 'ranking_date'])
        
        # Initialize Elo system
        elo_dict = {}
        K = 32
        default_elo = 1500
        results = []
        
        # Get unique dates with sufficient data
        date_counts = master['ranking_date'].value_counts().sort_index()
        valid_dates = date_counts[date_counts >= 100].index
        unique_dates = sorted(valid_dates)
        
        if not unique_dates:
            raise ValueError("No valid dates with sufficient data found")
        
        print(f"\n📈 Processing {len(unique_dates)} weeks of rankings...")
        
        # Initialize progress bar
        pbar = tqdm(unique_dates, desc="Calculating Elo ratings", unit="week")
        
        # Pre-cache player histories
        print("🔍 Caching player histories...")
        player_histories = {}
        for pid in tqdm(master['player_id'].unique(), desc="Caching", unit=" players"):
            player_histories[pid] = master[master['player_id'] == pid]
        
        for date in pbar:
            current_rankings = master[master['ranking_date'] == date].sort_values('ranking')
            
            # Initialize Elo for new players
            for pid in current_rankings['player_id'].unique():
                if pid not in elo_dict:
                    elo_dict[pid] = default_elo
            
            # Process rankings in chunks for better performance
            chunk_size = 1000
            for i in range(0, len(current_rankings), chunk_size):
                chunk = current_rankings.iloc[i:i+chunk_size]
                
                for _, row in chunk.iterrows():
                    pid = row['player_id']
                    current_elo = elo_dict[pid]
                    current_rank = row['ranking']
                    
                    # Get previous ranking
                    prev_rank = None
                    hist = player_histories[pid]
                    prev_rankings = hist[hist['ranking_date'] < date]
                    
                    if not prev_rankings.empty:
                        prev_rank = prev_rankings.iloc[-1]['ranking']
                    
                    # Calculate result (1=win, 0=loss, 0.5=draw)
                    if prev_rank is not None:
                        if current_rank < prev_rank:
                            result = 1
                        elif current_rank > prev_rank:
                            result = 0
                        else:
                            result = 0.5
                    else:
                        result = 0.5
                    
                    # Find nearby players for Elo calculation
                    nearby = current_rankings[
                        (current_rankings['ranking'] >= current_rank - 10) & 
                        (current_rankings['ranking'] <= current_rank + 10) &
                        (current_rankings['player_id'] != pid)
                    ]
                    
                    if len(nearby) > 0:
                        opp_elos = [elo_dict[p] for p in nearby['player_id']]
                        avg_opp_elo = sum(opp_elos) / len(opp_elos)
                        expected = 1 / (1 + 10 ** ((avg_opp_elo - current_elo) / 400))
                    else:
                        expected = 0.5
                    
                    # Update Elo
                    elo_change = K * (result - expected)
                    new_elo = current_elo + elo_change
                    elo_dict[pid] = new_elo
                    
                    results.append({
                        'player_id': pid,
                        'ranking_date': date,
                        'ranking': current_rank,
                        'elo_rating': new_elo,
                        'elo_change': elo_change
                    })
            
            # Update progress bar
            pbar.set_postfix({
                'Players': len(elo_dict),
                'Current Date': date.strftime('%Y-%m-%d')
            })
        
        # Convert results to DataFrame
        elo_df = pd.DataFrame(results)
        
        # Get latest Elo for each player
        latest_elo = elo_df.sort_values('ranking_date').groupby('player_id').tail(1)
        
        # Prepare current rankings for merge
        current['player'] = current['player'].astype('int32')
        latest_elo['player_id'] = latest_elo['player_id'].astype('int32')
        
        # Merge with current rankings
        merged = current.merge(
            latest_elo[['player_id', 'elo_rating']], 
            left_on='player', 
            right_on='player_id', 
            how='left'
        )
        merged['elo_rating'] = merged['elo_rating'].fillna(default_elo)
        
        # Add player names if available
        try:
            players = pd.read_csv("players_detail.csv", 
                                usecols=['player_id', 'name_first', 'name_last'],
                                dtype={'player_id': 'int32'})
            merged = merged.merge(
                players,
                left_on='player',
                right_on='player_id',
                how='left'
            )
            merged['player_name'] = merged['name_first'] + ' ' + merged['name_last']
            merged = merged.drop(['name_first', 'name_last'], axis=1)
        except Exception as e:
            print(f"⚠️ Could not add player names: {str(e)}")
            merged['player_name'] = 'Unknown'
        
        # Calculate additional metrics
        merged['elo_rank'] = merged['elo_rating'].rank(ascending=False, method='min').astype('int32')
        merged['rank_diff'] = merged['rank'] - merged['elo_rank']
        merged['performance'] = np.select(
            [
                merged['rank'] < merged['elo_rank'],
                merged['rank'] > merged['elo_rank']
            ],
            ['Overperforming', 'Underperforming'],
            default='On Par'
        )
        
        # Final output columns
        output_cols = [
            'player', 'player_name', 'rank', 'elo_rank', 'rank_diff',
            'points', 'elo_rating', 'performance', 'ranking_date'
        ]
        output_cols = [col for col in output_cols if col in merged.columns]
        merged = merged[output_cols].sort_values('rank')
        
        # Save to CSV
        output_file = "atp_current_with_elo.csv"
        merged.to_csv(output_file, index=False)
        
        # Calculate and display statistics
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Successfully processed {len(merged):,} players in {duration:.1f} seconds")
        print(f"📊 Elo ratings range from {merged['elo_rating'].min():.0f} to {merged['elo_rating'].max():.0f}")
        
        # Display top players
        print("\n🏆 Top 10 Players by Elo Rating:")
        print(merged.nsmallest(10, 'elo_rank')[
            ['rank', 'elo_rank', 'player_name', 'elo_rating']
        ].to_string(index=False))
        
        return merged
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    calculate_elo_ratings()
