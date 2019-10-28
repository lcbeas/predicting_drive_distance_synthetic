# predicting_drive_distance_synthetic
Created realistic synthetic data on drive distance with player and hole as factors. Predict without referencing true values, then compare to see accuracy of predictions. 

Workflow: fixed_effects_regression.py --> FE_regression.R --> FE_regression_analysis.py

Use inputs in fixed_effects_regression.py to create realistic synthetic data on golf shots. Each row of the data frame represents a shot with features of distance, player, and hole. 

Export file with data frame to FE_regression.R to run large fixed effects regression. * R has an extremely simple and intuitive way to do this with large data

Export results from R back to FE_regression_analysis.py to clean data and return RMSE. RMSE shows how accurately the model predicts the true values from the synthetic data we created. 
