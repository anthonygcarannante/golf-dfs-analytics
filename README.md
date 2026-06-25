# golf-dfs-analytics

# Golf DFS Analytics & Lineup Optimizer

A machine learning pipeline for predicting PGA Tour player performance and optimizing DraftKings golf lineups using strokes gained data.

## Overview

This project builds an end-to-end data science pipeline that:
1. Predicts whether a PGA Tour player will make the cut (classification)
2. Forecasts a player's DraftKings fantasy points (regression)
3. Optimizes a 6-player DraftKings lineup within a $50,000 salary cap (linear programming)

## Data

Historical PGA Tour data (2015-2022) sourced from Kaggle, including:
- Strokes gained metrics (putting, approach, around the green, off the tee)
- Tournament and course information
- DraftKings scoring and salary data
- Cut results and finish positions

## Models

### Cut Line Prediction (Classification)
- **Model:** Gradient Boosting Classifier
- **Features:** Strokes gained categories, course, tournament, purse, rounds
- **Performance:** 81% accuracy, ROC-AUC 0.891
- **Key finding:** Approach play (sg_app) and putting (sg_putt) are the strongest predictors of making the cut, accounting for 58% of combined feature importance

### DraftKings Points Forecasting (Regression)
- **Model:** Gradient Boosting Regressor
- **Features:** Strokes gained categories, course, tournament, purse, rounds
- **Performance:** MAE 10.47 DKP points, R² 0.505
- **Key finding:** Course context explains ~14% of predictive power beyond individual skill metrics, suggesting course fit adjustments meaningfully improve projections

## Lineup Optimizer

Uses linear programming (PuLP) to select 6 players that maximize projected DraftKings points within a $50,000 salary cap constraint. Includes a minimum projection floor to avoid rostering low-upside players.

```python
lineup, total_salary, total_projected = optimize_lineup(
    field_df, 
    salary_cap=50000, 
    roster_size=6, 
    min_projection=60
)
```

## Project Structure
golf_analytics/
│
├── data/
│   └── pga_tour_data.csv          # Raw dataset
│
├── notebooks/
│   └── golf_analytics.ipynb       # Main analysis notebook
│
├── models/
│   ├── cut_prediction_model.pkl   # Saved cut line classifier
│   └── dkp_regression_model.pkl   # Saved DKP forecasting model
│
└── README.md

## Key Findings

- **Putting and approach play** are the primary determinants of both making the cut and scoring DraftKings points on the PGA Tour
- **Course context** adds meaningful signal beyond individual skill — particularly for DKP prediction where course importance (0.105) rivals individual strokes gained categories
- **Field strength** (proxied by purse size) improves DKP prediction, suggesting tournament prestige affects scoring patterns
- **Made cut players only** should be used for DKP optimization — the bimodal distribution of fantasy points reflects the fundamental difference between 2-round and 4-round scoring

## Future Improvements

- Integrate live DraftKings salary data via web scraping
- Add current season strokes gained stats via DataGolf API
- Incorporate historical player-course performance as a feature
- Add ownership projections to identify leverage plays
- Backtest optimizer performance against historical contest results

## Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn (Gradient Boosting)
- PuLP (Linear Programming)
- Matplotlib
- Kaggle Notebooks
