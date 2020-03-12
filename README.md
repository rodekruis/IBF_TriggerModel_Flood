# IBF_TriggerModel_Flood V1.1

Initial version of Trigger Model methodology for floods, using global datasets. The objetive is to assess the predictability of historical floods impact (YES/NO)  recorded per district in a country, using global datasets.

VERSION V1.1.1 : Using only global discharge input data (Glofas) and testing a set of fixed discharged thresholds per glofas virtual stations.

VERSION V1.1.2 :  Using only global discharge input data (Glofas) but with simple decision tree to test which threshold migh be better to use.

## Directory Structure
-   `scripts` model and visualization scripts
-   `africa` global input data for all Africa
-   `uganda`, `kenya`... input and output data per country

## Setup

#### Requirements:
-   [Python 3.7.4](https://www.python.org/downloads/)

to install necessary modules, execute
```bash
pip install -r requirements.txt
```

## Model

### What does it do?

1. extract Glofas virtual stations  

1. compute 3 discharge thresholds based on glofas discharge quantiles Q50, Q80 and Q90 for each station. These quantiles correspond respectively to return periods of 2, 10 and 20 years. These quantiles are calculated based on yearly extreme event analysis.  

1. plot the historical glofas discharge per district for the selected relevant stations per district  

1. compute the performance of the model per district for the 3 specific Quantile (Q50, Q80, Q90) and save all results in a .CSV 

### How do I execute it?

to run the model, execute
```
python scripts/V112_glofas_analysis_refactor.py
```
`V112_glofas_analysis_refactor.py` accepts the command line arguments described below,

```
usage: V112_glofas_analysis_refactor.py [-h]
                                        [country] [ct_code] [model] [loss]

positional arguments:
  country     [Uganda]
  ct_code     [uga]
  model       [quantile]
  loss        [far]

optional arguments:
  -h, --help  show this help message and exit

```

## Visualization

### How do I visualize model performance?
to visualize performance, execute 
```
python scripts/IBF_flood_model_performance_visual.py
```
this will create maps of the performance or the model per district, by plotting FAR, POD, POFD, CSI and the number of available events per district. 
