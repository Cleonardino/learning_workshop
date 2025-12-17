## Mandatory steps
Data preparation
Visualisation
Feature extraction
Modelling
Evaluation

## Data preparation
Interpolation of meteorological data because energy is each minute, meteo is each hour
Some energy minutes data are missing, we duplicate previous data to preserve jump of energy to have on and off preserved.

## Visualisation
Create visuals to understand more precisely the data and its context :
- Correlation matrix
- Data histograms (values repartition)
- Data values (time as abscissa)

## Feature extraction
Create new columns for improved models later on :
- Day of week
- Is weekend
- Season
- Is public holiday
- Is business hour

## Modelling
We each try a different model to see which one works the best

### Classify and Regress (Clément Desberg)