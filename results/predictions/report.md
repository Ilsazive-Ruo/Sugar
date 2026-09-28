# New sample predictions

- Input: `ToBeScreened.csv`
- Model directory: `models`
- Input rows: 600
- Predictions: 600

## Top Predictions By Feature Set

| feature_set | rank | Sugar name | Concentration (% w/v) | predicted_Luciferase expression (RLU) |
| --- | --- | --- | --- | --- |
| provided_plus_rdkit_descriptors | 1 | α-D-glucose | 6 | 1.03536e+07 |
| provided_plus_rdkit_descriptors | 2 | D-Glucose | 6 | 1.03488e+07 |
| provided_plus_rdkit_descriptors | 3 | β-D-Glucose | 6 | 9.93964e+06 |
| provided_plus_rdkit_descriptors | 4 | D-Glucose | 9 | 9.61135e+06 |
| provided_plus_rdkit_descriptors | 5 | Sucrose | 6 | 9.22933e+06 |
| provided_plus_rdkit_descriptors | 6 | D-Psicose | 6 | 9.00087e+06 |
| provided_plus_rdkit_descriptors | 7 | α-D-glucose | 9 | 8.99822e+06 |
| provided_plus_rdkit_descriptors | 8 | α-D-Glucose-1-phosphate disodium  | 6 | 8.7918e+06 |
| provided_plus_rdkit_descriptors | 9 | β-D-Glucose | 9 | 8.668e+06 |
| provided_plus_rdkit_descriptors | 10 | Sucrose | 3 | 8.58168e+06 |

## Output Files

- `results\predictions\predictions.csv`: predictions
- `results\predictions\predictions_ranked.csv`: predictions_ranked
- `results\predictions\smiles_cleaning_report.csv`: cleaning
- `results\predictions\prediction_profile.json`: profile
