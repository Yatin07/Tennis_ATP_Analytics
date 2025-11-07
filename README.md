# 🎾 Tennis ATP Analytics & Prediction Dashboard

A Power BI and Python integrated dashboard that visualizes and predicts ATP tennis player performance using machine learning and real-world data.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/Yatin07/Tennis_ATP_Analytics?style=social)](https://github.com/Yatin07/Tennis_ATP_Analytics/stargazers)

A comprehensive data analysis and machine learning project that evaluates and forecasts tennis player performance using real-world ATP ranking data. This dashboard serves as a unified analytical platform, combining historical performance visualization with AI-driven match prediction, helping users understand player comparisons in head-to-head scenarios.

## 📋 Directory Structure

```
Tennis_Project/
│
├── XG Boost Model/               # XGBoost model and training scripts
│   ├── feature_importance.png   # Visualization of feature importance
│   ├── model_metadata.json      # Model configuration and metadata
│   ├── tennis_model_full.json   # Pre-trained XGBoost model
│   ├── train_full_model.py      # Script to train the full model
│   ├── train_tennis_model.py    # Main training script
│   ├── train_tennis_model_debug.py  # Debug version of training script
│   └── training_timing.json     # Training performance metrics
│
├── csv/                         # Source data in CSV format
│   ├── atp_rankings_current.csv  # Current ATP rankings
│   ├── flags_iso (1).csv        # Country flag data for visualization
│   ├── matches_master.csv       # Historical match data
│   ├── players_detail.csv       # Player information and metadata
│   └── rankings_master.csv      # Historical ranking data
│
├── elo files/                   # ELO rating calculation scripts and data
│   ├── atp_current_with_elo.csv # Pre-calculated ELO ratings
│   ├── calculate_elo_fast.py    # Optimized ELO calculation
│   └── calculate_elo_ratings.py # Standard ELO calculation
│
├── parquet/                     # Data in Parquet format
│   ├── matches_master.parquet   # Match data (Parquet format)
│   └── rankings_master.parquet  # Ranking data (Parquet format)
│
├── Tennis_ATP_Analytics.pbix    # Power BI dashboard file
└── README.md                    # Project documentation
```

## 📋 Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Introduction](#introduction)
- [Data Sources](#data-sources)
- [Data Loading and Preparation](#data-loading-and-preparation)
- [Data Transformation](#data-transformation)
- [Data Modeling](#data-modeling)
- [DAX Calculations](#dax-calculations)
- [Data Visualization](#data-visualization)
- [Elo CSV Creation Process](#elo-csv-creation-process)
- [XGBoost Model Creation](#xgboost-model-creation)
- [Project Questions and Deliverables](#project-questions-and-deliverables)

## Setup and Installation

### Prerequisites

1. *Power BI Desktop*
   - Download and install the latest version of [Power BI Desktop](https://powerbi.microsoft.com/en-us/desktop/)
   - Ensure you have administrator privileges for installation

2. *Python Environment*
   - Install [Python 3.8 or later](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify installation by running in Command Prompt:
     
     python --version
     pip --version
     

3. *Required Python Packages*
   Install the following packages using pip:
   bash
   pip install pandas numpy xgboost scikit-learn matplotlib requests tqdm
   

### Project Setup

1. *Clone the Repository*
   bash
   git clone https://github.com/Yatin07/Tennis_ATP_Analytics.git
   cd Tennis_ATP_Analytics
   

2. *Download Pre-Processed Data and Model*
   - The repository already includes all necessary pre-processed files:
     - `csv/matches_master.csv`: Combined historical match data
     - `csv/rankings_master.csv`: Historical ranking data
     - `csv/players_detail.csv`: Player information
     - `csv/atp_rankings_current.csv`: Latest ATP rankings
     - `elo files/atp_current_with_elo.csv`: Pre-calculated Elo ratings
     - `XG Boost Model/tennis_model_full.json`: Pre-trained XGBoost model
     - `XG Boost Model/model_metadata.json` & `XG Boost Model/feature_importance.png`: Model details
   - No additional data processing is required as all files are ready to use

### Power BI Setup

1. *Install Required Software*
   - Download and install [Power BI Desktop](https://powerbi.microsoft.com/en-us/desktop/)
   - Install [Python 3.8 or later](https://www.python.org/downloads/)
     - During installation, check "Add Python to PATH"

2. *Install Required Python Packages*
   Open Command Prompt and run:
   bash
   pip install pandas numpy xgboost scikit-learn matplotlib requests tqdm
   

3. *Configure Python in Power BI*
   - Open Power BI Desktop
   - Go to File > Options and settings > Options > Python scripting
   - Set the Python home directory to your Python installation path (e.g., C:\Users\YourUsername\AppData\Local\Programs\Python\Python38)
   - Restart Power BI after configuration

4. *Update Data Source Paths*
   - After opening Tennis_ATP_Analytics.pbix, you'll see a warning about data source errors
   - Click "Transform data" > "Data source settings"
   - For each data source (CSV file):
     1. Select the file
     2. Click "Change Source..."
     3. Update the file path to point to the file's location on your system
     4. Click "OK"
   - Click "Close & Apply" to save changes

5. *Configure Python Visual*
   - In the dashboard, locate the prediction visual (usually shows match predictions)
   - Click on the visual to select it
   - In the Python script editor (bottom panel), locate the model loading code:
     python
     model_path = r'C:\path\to\XG Boost Model\tennis_model_full.json'  # Update this path
     model = xgb.Booster()
     model.load_model(model_path)
     
   - Update the model_path to point to the tennis_model.json file in your local repository
   - Click the play button (▶) to apply changes

6. *Refresh Dashboard*
   - Click "Refresh" in the Home tab to update all visuals
   - The dashboard should now load with your local data and model

### Using the Pre-Trained Model

The dashboard includes Python visuals that use the pre-trained XGBoost model (tennis_model_full.json). The model is already configured to work with the provided data files. No additional training is required.

To use the prediction features in Power BI:
1. Ensure all files from the repository are kept in their original locations
2. The dashboard will automatically load the pre-trained model when opened
3. The prediction visuals will work out-of-the-box with the included data

### Common Issues and Solutions

1. *Data Source Errors*
   - If you see "The key didn't match any rows in the table" errors:
     - Ensure all CSV files are in the correct location
     - Verify file paths in Power Query Editor (right-click query > Advanced Editor)
     - Check that file names match exactly (case-sensitive)

2. *Python Visual Not Working*
   - If the prediction visual shows an error:
     - Verify Python is properly configured in Power BI
     - Check that all required packages are installed
     - Ensure the model path in the Python script is correct
     - Look for error messages in the Python script editor

3. *Performance Issues*
   - For better performance with large datasets:
     - Close other applications
     - Increase Power BI's memory limit in File > Options > Global > Memory
     - Consider using Power BI's Performance Analyzer to identify slow visuals

### Troubleshooting

1. *Python Package Issues*
   bash
   # If you encounter package conflicts
   pip install --upgrade package-name
   
   # Or create a virtual environment
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   

2. *Power BI Python Script Errors*
   - Verify Python path in Power BI settings
   - Ensure all required packages are installed in the Python environment used by Power BI
   - Check the Python script output in Power BI's Python script editor for detailed error messages

3. *Data Loading Issues*
   - Verify file paths in the scripts
   - Ensure all required CSV files are in the correct directory
   - Check file permissions and encoding



---

## Introduction

The *Tennis Player Performance Analysis & Prediction Dashboard* is an interactive data analytics project that evaluates and forecasts tennis player performance using real-world ATP ranking data. This dashboard serves as a unified analytical platform, combining historical performance visualization with AI-driven match prediction, helping users understand player comparisons in head-to-head scenarios.

Developed with Power BI for visualization and Python integration for predictive modeling, this project allows users to explore player trends, strengths, and consistency over time. The prediction module leverages statistical and machine learning models to simulate match outcomes, offering data-backed insights based on ranking points, previous matches, and performance metrics.

### Key Features
- *Interactive Visualizations:* Intuitive visuals and user-friendly filters designed following HCI principles.
- *Predictive Analytics:* AI-driven match prediction using historical data and machine learning models.
- *Comprehensive Insights:* Combines data visualization, predictive analytics, and performance monitoring.

## Data Sources

The datasets used in this project are sourced from the [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) repository on GitHub. This repository provides comprehensive ATP match and ranking data essential for analysis and prediction. We are specifically using the ATP matches YYYY.csv files for our visualization purposes.

## Data Loading and Preparation

### Data Collection and Preparation Process
1. *Match Data Processing:*
   - Combined yearly ATP match files (atp_matches_YYYY.csv) into a single dataset using Python scripts.
   - Added a year column from filenames and removed invalid records.
   - Saved as matches_master.csv.

2. *Ranking Data Processing:*
   - Merged all ATP ranking snapshots into rankings_master.csv using Python scripts not merged the atp_rankings_current.csv.
   - Standardized formats and ensured consistency.

3. *Current Rankings:*
   - Used the `csv/atp_rankings_current.csv` directly from GitHub and cleaned it for analysis.

4. *Player Information:*
   - Utilized player metadata with Wikidata IDs to attempt photo extraction via Python scripts.
   - Created an image URL column for players, although not all players have images due to missing Wikidata IDs.
   - *Warning:* Not all images may be accurate or valid, as they are sourced from various locations.

5. *Flag Data Integration:*
   - Incorporated `csv/flags_iso (1).csv` containing country flag URLs to enhance visuals in Power BI.

6. *Data Integration:*
   - Linked match data with player rankings and ensured temporal alignment.

7. *Quality Assurance:*
   - Verified date ranges, checked for inconsistencies, and validated player references.

### Final Output
- `csv/matches_master.csv`: Complete historical match data.
- `csv/rankings_master.csv`: Historical ranking data.
- `csv/atp_rankings_current.csv`: Latest ATP rankings.
- `csv/players_detail.csv`: Player information with image URLs.
- `elo files/atp_current_with_elo.csv`: Player rankings with calculated ELO ratings.

## Data Transformation

In this stage, the raw datasets were systematically cleaned, merged, and enhanced to create an analysis-ready structure for both visual analytics and player performance prediction. The main objective was to transform fragmented match and player data into a single, enriched dataset that could be efficiently used in Power BI visuals and Python-based prediction models.

### Key Transformation Steps

1. *Merging Player Details with Match Records:*
   - The matches_master table was joined twice with the players_detail table — once on winner_id and once on loser_id.
   - This enriched every match record with both players' details such as first and last name, age, height, hand orientation, and country.
   - The expanded output, referred to as Merge1, became the core dataset for later modeling and visualization.

2. *Feature Engineering and Data Cleaning:*
   - Within Power BI's Python environment, the merged dataset was refined using a Random Forest-based script.
   - Numeric fields were converted and cleaned.
   - New analytical features were created such as rank_diff, age_diff, height_diff, and is_five_setter.
   - Surface type was one-hot encoded, and rounds were label-encoded for model compatibility.
   - This prepared dataset was then split into training and testing subsets to validate model accuracy.

3. *Integration of Flag Data for Visual Appeal:*
   - To make the dashboard more interactive and engaging, a flag.csv file containing each player's country flag URL was imported.
   - This file was connected to both Merge1 and players_detail tables using the country field as a key.
   - It allowed Power BI visuals to dynamically display player flags beside names, making nationality-based comparisons clearer and more visually appealing.

### Purpose and Importance

These transformation steps ensured data consistency, reduced redundancy, and allowed the creation of advanced performance metrics. The final transformed dataset acted as a unified analytical model, linking player identity, match performance, and country context — enabling both descriptive insights (rank trends, win rates, surfaces) and predictive insights (head-to-head outcomes).

## Data Modeling

### Schema Type
- *Galaxy Schema (Fact Constellation):* Multiple fact tables connected to a common dimension table, Dim_players_detail.

### Table Classification
- *Dimension Tables:*
  - Dim_players_detail: Central reference for player attributes.
  - flags_iso: Holds flag URLs for visual enrichment.

- *Fact Tables:*
  - Fact_matches_master: Historical match data.
  - Fact_rankings_master: Historical ranking timeline.
  - Fact_atp_rankings_current: Current ATP rankings.
  - Fact_PlayerMatches: Player-centric match records.
  - Fact_Merge1: Combined dataset for analytics.

## DAX Calculations

### Fact_PlayerMatches

The Fact_PlayerMatches table was created in Power BI using a DAX UNION of the Fact_matches_master table to represent each player's perspective in every match — both winner and loser — as individual records.

*Purpose and Explanation*

- *Why It Was Created:*
  - The original Fact_matches_master table stores matches at the event level, with each row representing a single match with two participants (winner and loser).
  - To enable player-level insights, such as the number of matches played, won, or lost, and win percentage across surfaces, a player-centric fact table was required — hence, Fact_PlayerMatches.

- *How It Works:*
  - The DAX query uses UNION to combine two derived tables:
    - *Winner perspective:* Selects the winner_id, winner_name, sets is_win = TRUE(), and includes the match surface.
    - *Loser perspective:* Selects the loser_id, loser_name, sets is_win = FALSE(), and also includes the surface.
  - Each match is split into two records, one for each player's outcome, allowing for player-level aggregation.

## Data Visualization

### Dashboards
1. *Current Global Tennis Rankings Insights Dashboard:*
   - Provides an overview of ATP player statistics and rankings.
   - Key insights include KPI cards, scatter plots, and country-wise player counts.

2. *ATP Player Insights Dashboard:*
   - Focuses on individual player analytics and performance profiles.
   - Features player search, ranking comparisons, and career trend analysis.

3. *ATP Player Performance Analysis Dashboard:*
   - Analyzes match-level performance across surfaces and opponents.
   - Includes player selectors, KPI cards, and surface specialization analysis.

### Python Visuals
1. *XGBoost-Based Prediction Visual*
   
   *Purpose:*
   - This Python visual leverages a pre-trained XGBoost model (tennis_model.json) to predict the winning probability between two selected players based on their historical performance attributes.
   
   *Key Steps Performed:*
   - *Imported Libraries:* Utilizes pandas, xgboost, and matplotlib for data manipulation, model loading, and visualization.
   - *Model Loading:*
     python
     model = xgb.Booster()
     model.load_model(model_path)
     
   - *Data Preparation:*
     - Cleaned and encoded match-level data, including ranking, age, height, and surface type.
     - Used slicer-selected players (playerA and playerB) to fetch their recent statistics.
   - *Feature Vector Construction:*
     python
     features = pd.DataFrame([{
         'rank_diff': playerB_rank - playerA_rank,
         'age_diff': playerA_age - playerB_age,
         'ht_diff': playerA_height - playerB_height,
         'surface_encoded': surface_type
     }])
     
   - *Prediction and Visualization:*
     - Predicted win probability using the model.
     - Visualized the result as a horizontal bar chart showing:
       - Predicted winner name
       - Confidence percentage
       - Winning probability of each player
   
   *Example Insight:*
   - If Rafael Nadal and Roger Federer are selected, the visual might show:
     - 🎾 Predicted Winner: Nadal (78.6% Confidence) on Hard Court
   - This visual demonstrates how machine learning can be embedded within Power BI for real-time AI predictions directly inside an analytical dashboard.

2. *Simple Head-to-Head Record Visual*

   *Purpose:*
   - This visual performs a historical win comparison between two selected players, providing an intuitive chart showing how many times each player has defeated the other in recorded ATP matches.

   *Process Overview:*
   - *Data Filtering:*
     - Filters the dataset using slicer inputs to detect the two selected players.
   - *Win Count Calculation:*
     python
     p1_wins = len(df[(df['winner_name'] == player1) & (df['loser_name'] == player2)])
     p2_wins = len(df[(df['winner_name'] == player2) & (df['loser_name'] == player1)])
     
   - *Visualization:*
     - Displays results through a bar chart comparing total wins for each player.
     - Adds labels, colors, and warnings for unmatched pairs (if players have never faced each other).

   *Example Output:*
   - A bar chart showing:
     - Novak Djokovic — 30 Wins
     - Andy Murray — 11 Wins
   - This helps users visualize rivalry dominance and compare players' historical performances visually and interactively.

🎯 *Outcome*
- Both visuals together provide:
  - *Descriptive analytics:* Actual win–loss records.
  - *Predictive analytics:* AI-based future match outcomes.
- By combining Power BI slicers, Python scripting, and XGBoost machine learning, the H2H Dashboard becomes a powerful analytical tool for real-time player comparison and prediction.

## Elo CSV Creation Process

The Elo CSV provides an alternative ranking metric for player performance analysis. Although not currently utilized in the project, it is available for further development or analysis.

### Data Loading
- *Historical Rankings:* Loaded from rankings_master.csv.
- *Current Rankings:* Loaded from atp_rankings_current.csv.

### Data Preprocessing
- Converted ranking_date to datetime format and handled missing values.
- Sorted data by player_id and ranking_date for accurate processing.

### Elo Rating System Initialization
- *Elo Dictionary:* Initialized with a default rating of 1500 for new players.
- *K-factor:* Set to 32, determining the sensitivity of rating changes.

### Elo Calculation
- Iterated over unique ranking dates to update player Elo ratings.
- Calculated match results (win, loss, draw) based on ranking changes.
- Determined expected scores against average opponents using the Elo formula:
  
  \[
  \text{Expected} = \frac{1}{1 + 10^{\left(\frac{\text{Opponent Elo} - \text{Current Elo}}{400}\right)}}
  \]

- Updated Elo ratings based on match results and expected scores.

### Data Merging
- Merged latest Elo ratings with current rankings.
- Added player names from players_detail.csv if available.

### Additional Metrics
- Calculated elo_rank, rank_diff, and performance indicators (Overperforming, Underperforming, On Par).

### Output
- Saved the merged data with Elo ratings to atp_current_with_elo.csv.

This Elo CSV can be leveraged to provide deeper insights into player performance, offering a robust tool for those interested in exploring alternative ranking systems.

## XGBoost Model Creation

The XGBoost model used in this project was created to predict tennis match outcomes based on historical performance data. Here is an overview of the model creation process:

### Data Loading and Preprocessing
- *Data Source:*
  - Loaded from matches_master.csv.
- *Feature Engineering:*
  - Created features: rank_diff, age_diff, ht_diff, and surface_encoded.
  - Prepared a balanced dataset by duplicating data with swapped players to ensure both win/loss classes are represented.

### Feature Encoding
- *Surface Encoding:*
  - Surface types were encoded using a LabelEncoder for model compatibility.

### Model Training
- *Data Splitting:*
  - Split the data into training and testing sets.
- *DMatrix Creation:*
  - Created XGBoost DMatrix for both training and testing data.
- *Training Parameters:*
  - Used parameters such as objective: binary:logistic, max_depth, learning_rate, and others for optimal performance.
- *Cross-Validation:*
  - Performed cross-validation to evaluate model performance.

### Model Evaluation
- *Metrics:*
  - Evaluated using accuracy and ROC AUC.
- *Feature Importance:*
  - Assessed and visualized feature importance to understand model decisions.

### Model Saving
- *Format:*
   - Saved the trained model in JSON format as tennis_model_full.json.
- *Metadata:*
   - Saved metadata including feature names and encoding details in model_metadata.json (located in XG Boost Model/).

### Feature Importance Plot
- *Visualization:*
   - Generated a plot of feature importance saved as feature_importance.png (located in XG Boost Model/).

This comprehensive approach ensures the model is robust and ready for predictive analysis, leveraging historical data to forecast future match outcomes effectively.

## Project Questions and Deliverables

### Key Questions
- How do ATP players' rankings fluctuate over time?
- Which players dominate specific surfaces and tournaments?
- Which countries produce top-ranked players?
- How do two players compare head-to-head?
- Can machine learning accurately predict match outcomes?

### Deliverables
1. *Cleaned Datasets:* Processed CSV files for analysis.
2. *Data Model:* Galaxy schema for optimized reporting.
3. *Interactive Dashboard:* Visual insights on player trends and comparisons.
4. *Python Integration:* Embedded visuals for historical and predictive analysis.
5. *Machine Learning Model:* Pre-trained model for win probability prediction.
6. *Documentation Report:* Detailed methodology and insights.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Jeff Sackmann](https://github.com/JeffSackmann) for the comprehensive tennis data
- Open-source contributors to the data science and tennis analytics community
- The ATP Tour for making match data publicly available

---