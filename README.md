# IBF_TriggerModel_Flood V1.1

Latest version of Trigger Model methodology for floods, using global datasets. The objetive is to assess the predictability of historical floods impact (YES/NO)  recorded per district in a country, using global datasets.

VERSION V1.1.1 : Using only global discharge input data (Glofas) and testing a set of fixed discharged thresholds per glofas virtual stations.

VERSION V1.1.2 :  Using only global discharge input data (Glofas) but with simple decision tree to test which threshold migh be better to use.

On the folder /scripts, two python scripts need to be run :  

V111_glofas_analysis.py :  

- extract Glofas virtual stations  

- compute 3 discharge thresholds based on glofas discharge quantiles Q50, Q80 and Q90 for each station. These quantiles correspond respectively to return periods of 2, 10 and 20 years. These quantiles are calculated based on yearly extreme event analysis.  

- plot the historical glofas discharge per district for the selected relevant stations per district  

- compute the performance of the model per district for the 3 specific Quantile (Q50, Q80, Q90 and save all results in a .CSV  

IBF_flood_model_performance_visual.py : a script to create maps of the performance or our model per district.This is plotting FAR, POD, POFD, CSI and the number of available events per district. This is also showing the coverage of our prediction model ! 
