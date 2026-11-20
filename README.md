# Titanic Passenger Survival Study

A beginner-friendly Python exploratory data analysis project studying survival patterns among Titanic passengers.

## Objectives
 Inspect passenger data and missing values.
 Calculate overall survival counts and percentage.
 Compare survival by sex and passenger class.
 Explore age, fare, and family-size patterns.
 Generate charts and save cleaned data and summary tables.

## Dataset
Use the Titanic training dataset (`train.csv`) from the Kaggle Titanic competition: https://www.kaggle.com/competitions/titanic/data . Rename the downloaded file to `titanic.csv` and place it inside the `data/` folder. The training file contains the `Survived` target column. Check the dataset's terms before redistributing the raw CSV; you can keep it out of your public repository and include only the source link.

## Setup (Windows / VS Code)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/analyze_titanic.py
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## Outputs
The script creates cleaned data, summary CSV files, and PNG charts in `outputs/` and `outputs/charts/`.
