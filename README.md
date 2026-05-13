# Global Weather Trend Forecasting

## Overview
This project analyzes the **GlobalWeatherRepository.csv** dataset containing daily weather information for 257 cities across 211 countries, with 41 features covering temperature, precipitation, air quality, wind, visibility, and astronomical data.

The analysis spans basic and advanced data science techniques including data cleaning, exploratory data analysis, forecasting, anomaly detection, climate analysis, environmental impact analysis, feature importance, and spatial/geographical pattern analysis.

## Dataset
- **Rows**: 140,728
- **Features**: 41 (temperature, precipitation, humidity, pressure, wind, air quality, UV index, visibility, astronomical)
- **Time Range**: Multiple daily observations across global locations
- **No missing values** in the dataset

## Project Structure

```
├── GlobalWeatherRepository.csv   # Dataset
├── analysis.py                   # Main analysis script
├── report.txt                    # Comprehensive text report
├── README.md                     # This file
├── eda_*.png                     # EDA visualizations (6 files)
├── forecast_*.png                # Forecasting results (2 files)
├── advanced_*.png                # Advanced EDA (2 files)
├── climate_*.png                 # Climate analysis (2 files)
├── env_*.png                     # Environmental impact (3 files)
├── feature_importance.png        # Feature importance chart
├── spatial_*.png                 # Spatial analysis maps (3 files)
└── geo_*.png                     # Geographical patterns (1 file)
```

## Methodology

### 1. Data Cleaning & Preprocessing
- Missing value check (none found)
- Outlier detection via IQR method with capping
- StandardScaler normalization for key features
- Temporal feature engineering (year, month, day, hour, dayofweek)
- Latitude band classification (Tropical, Temperate, Arctic/Antarctic)

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis (temperature, precipitation, humidity)
- Correlation matrix of 10 weather parameters
- Time series plots (temperature, precipitation, humidity over time)
- Country-level aggregations (hottest, coldest, wettest)
- Weather condition frequency analysis

### 3. Forecasting Models
- **Temporal split**: 80/20 train/test on daily aggregated data
- **Features**: Lag features (1, 2, 3, 7 days), rolling statistics (7, 30-day), temporal features
- **Models tested**:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Metrics**: MAE, RMSE, R²

### 4. Advanced EDA
- **Anomaly Detection**: Isolation Forest (5% contamination) and Z-score analysis
- **Stationarity Test**: Augmented Dickey-Fuller test on temperature
- **Seasonal Decomposition**: Additive model with 30-day period

### 5. Ensemble Forecasting
- Weighted average ensemble (RF + GB + XGBoost)
- Simple average ensemble
- Comparison against individual model performance

### 6. Climate Analysis
- Latitude band analysis (Tropical, Temperate, Polar)
- Seasonal patterns by hemisphere
- Temperature-latitude relationship
- Seasonal decomposition

### 7. Environmental Impact Analysis
- Air quality distribution (US EPA Index categories)
- Correlation between air pollutants and weather parameters
- Country-level pollution rankings (PM2.5)
- Pollutant vs weather scatter plots

### 8. Feature Importance
- Built-in importance from XGBoost, Random Forest, Gradient Boosting
- Average importance across 3 models
- Permutation importance for model-agnostic assessment

### 9. Spatial Analysis
- Global temperature map (latitude/longitude scatter)
- Global precipitation map
- Global PM2.5 air quality map

### 10. Geographical Patterns
- Continent-level weather aggregation
- Cross-continent comparison (temperature, precipitation, humidity, wind, air quality, UV)

## Results Summary

### Best Temperature Forecast
| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.37°C | 1.02°C | 0.77 |
| XGBoost | 0.59°C | 1.24°C | 0.67 |
| Gradient Boosting | 0.58°C | 1.25°C | 0.66 |
| Ensemble (Weighted) | 0.59°C | 1.25°C | 0.66 |

Linear Regression performed best overall due to the strong linear relationships in lagged temperature features.

### Key Insights
1. **Temperature** follows a clear latitudinal gradient with Tropical regions averaging 26°C and Temperate regions at 17°C
2. **Precipitation** is harder to forecast (max R² = 0.33) due to high variability
3. **Air Quality**: South Korea, India, and China have highest PM2.5 levels; Australia/Oceania shows lowest
4. **Ozone** correlates positively with temperature (photochemical formation)
5. **Wind speed** and **precipitation** help reduce particulate pollution
6. **Temperature lag-1** is the single most important predictor for forecasting
7. **Northern/Southern hemisphere** seasonal patterns are inverted as expected
8. **Seasonal decomposition** confirms strong annual cycles in global temperature

## Visualizations

| File | Description |
|------|-------------|
| `eda_distributions.png` | Distributions and boxplots of key variables |
| `eda_correlation.png` | Correlation heatmap of weather parameters |
| `eda_timeseries.png` | Global mean temperature, precipitation, humidity over time |
| `eda_country_temps.png` | Hottest and coldest countries |
| `eda_precip_country.png` | Wettest countries |
| `eda_conditions.png` | Most frequent weather conditions |
| `forecast_models_comparison.png` | Predicted vs actual for 5 models |
| `forecast_timeseries.png` | XGBoost forecast vs actual over time |
| `advanced_anomaly_detection.png` | Isolation Forest anomaly detection |
| `advanced_seasonal_decomp.png` | Seasonal decomposition of temperature |
| `climate_latitude_temp.png` | Temperature vs latitude |
| `climate_seasonal_patterns.png` | Seasonal patterns by hemisphere |
| `env_air_quality_dist.png` | Air quality category distribution |
| `env_air_quality_corr.png` | Air quality vs weather correlation |
| `env_pm25_vs_weather.png` | PM2.5 vs temperature, humidity, wind |
| `feature_importance.png` | Feature importance across 3 models |
| `spatial_temp_map.png` | Global temperature map |
| `spatial_precip_map.png` | Global precipitation map |
| `spatial_airquality_map.png` | Global PM2.5 map |
| `geo_continent_patterns.png` | Weather metrics by continent |

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, statsmodels, scipy
- xgboost

## Running the Analysis
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy xgboost
python analysis.py
```

## Deliverables
- `report.txt` — Comprehensive text report with all analyses, evaluations, and insights
- `README.md` — This documentation file
- 20+ PNG visualizations — Charts and maps supporting all analyses
- `analysis.py` — Reproducible Python script
