# SHAP and feature importance under repeated CV

- Data: `dataset.csv`
- Best model type from model comparison: `XGBoost`
- CV: 10 x 5-fold
- Feature sets: provided_only, provided_plus_rdkit_descriptors, rdkit_descriptors_only
- SHAP method: XGBoost pred_contribs

## Top Feature Importance

| feature_set | feature | mean_importance | std_importance | n_folds |
| --- | --- | --- | --- | --- |
| provided_only | Molecular weight (Da) | 0.332861 | 0.0294736 | 50 |
| provided_only | Solubility (mg/mL) | 0.311817 | 0.03067 | 50 |
| provided_only | Glass transition temperature (Tg, degC) | 0.186774 | 0.0175962 | 50 |
| provided_only | Concentration (% w/v) | 0.0928959 | 0.0200171 | 50 |
| provided_only | Viscosity (mPa.s) | 0.0756527 | 0.0100815 | 50 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA10 | 0.351293 | 0.0680153 | 50 |
| provided_plus_rdkit_descriptors | rdkit_Chi0v | 0.0462759 | 0.0388493 | 50 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState8 | 0.0309172 | 0.0105556 | 50 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState2 | 0.0273022 | 0.0215059 | 50 |
| provided_plus_rdkit_descriptors | rdkit_HallKierAlpha | 0.0267634 | 0.0183253 | 50 |
| provided_plus_rdkit_descriptors | rdkit_fr_Al_OH | 0.0252748 | 0.0403122 | 50 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState3 | 0.0229155 | 0.0193489 | 50 |
| provided_plus_rdkit_descriptors | rdkit_SlogP_VSA3 | 0.0221646 | 0.0225612 | 50 |
| provided_plus_rdkit_descriptors | rdkit_fr_aldehyde | 0.0188434 | 0.0422841 | 50 |
| provided_plus_rdkit_descriptors | rdkit_MaxPartialCharge | 0.0182144 | 0.00752905 | 50 |
| provided_plus_rdkit_descriptors | rdkit_SlogP_VSA2 | 0.0177886 | 0.019275 | 50 |
| provided_plus_rdkit_descriptors | rdkit_AvgIpc | 0.0151436 | 0.00614316 | 50 |
| provided_plus_rdkit_descriptors | rdkit_MolLogP | 0.0151328 | 0.00896196 | 50 |
| provided_plus_rdkit_descriptors | Solubility (mg/mL) | 0.0144184 | 0.00315296 | 50 |
| provided_plus_rdkit_descriptors | rdkit_PEOE_VSA8 | 0.0133579 | 0.0269094 | 50 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_CHGLO | 0.0124197 | 0.0100428 | 50 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState7 | 0.0111923 | 0.00749333 | 50 |
| provided_plus_rdkit_descriptors | rdkit_Phi | 0.0107559 | 0.0181211 | 50 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_LOGPHI | 0.0102543 | 0.0169731 | 50 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_CHGHI | 0.0102023 | 0.00632078 | 50 |
| rdkit_descriptors_only | rdkit_EState_VSA10 | 0.263388 | 0.0670362 | 50 |
| rdkit_descriptors_only | rdkit_fr_Al_OH_noTert | 0.119746 | 0.0607514 | 16 |
| rdkit_descriptors_only | rdkit_fr_Al_OH | 0.0829051 | 0.0409292 | 9 |
| rdkit_descriptors_only | rdkit_HallKierAlpha | 0.0676132 | 0.0246573 | 50 |
| rdkit_descriptors_only | rdkit_PEOE_VSA11 | 0.0572652 | 0.0213439 | 22 |
| rdkit_descriptors_only | rdkit_MolLogP | 0.0554612 | 0.0178077 | 50 |
| rdkit_descriptors_only | rdkit_NumHeteroatoms | 0.0542282 | 0.0410098 | 50 |
| rdkit_descriptors_only | rdkit_VSA_EState3 | 0.0520855 | 0.0101802 | 38 |
| rdkit_descriptors_only | rdkit_SMR_VSA1 | 0.0509416 | 0.0185418 | 50 |
| rdkit_descriptors_only | rdkit_NHOHCount | 0.0488786 | 0.0383797 | 49 |
| rdkit_descriptors_only | rdkit_SMR_VSA5 | 0.0478185 | 0.0100876 | 6 |
| rdkit_descriptors_only | rdkit_BCUT2D_CHGLO | 0.0453227 | 0.00328059 | 2 |
| rdkit_descriptors_only | rdkit_PEOE_VSA1 | 0.0431933 | 0.0112534 | 50 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | 0.0420485 | 0.0124301 | 31 |
| rdkit_descriptors_only | rdkit_TPSA | 0.0419359 | 0.0212128 | 50 |
| rdkit_descriptors_only | rdkit_Kappa2 | 0.0401156 | nan | 1 |
| rdkit_descriptors_only | rdkit_VSA_EState7 | 0.0368126 | 0.0140981 | 50 |
| rdkit_descriptors_only | rdkit_BCUT2D_CHGHI | 0.0362473 | nan | 1 |
| rdkit_descriptors_only | rdkit_BCUT2D_MRLOW | 0.0348982 | nan | 1 |
| rdkit_descriptors_only | rdkit_BCUT2D_LOGPLOW | 0.0302572 | 0.00715775 | 28 |

## Top SHAP

| feature_set | feature | mean_abs_shap | mean_shap |
| --- | --- | --- | --- |
| provided_only | Molecular weight (Da) | 965303 | -15413.8 |
| provided_only | Solubility (mg/mL) | 820797 | 21392.8 |
| provided_only | Glass transition temperature (Tg, degC) | 570088 | -18892.9 |
| provided_only | Viscosity (mPa.s) | 301327 | 978.264 |
| provided_only | Concentration (% w/v) | 251588 | 6819.85 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA10 | 614076 | 6470.63 |
| provided_plus_rdkit_descriptors | Solubility (mg/mL) | 539603 | -3685.32 |
| provided_plus_rdkit_descriptors | Concentration (% w/v) | 265074 | 2087.1 |
| provided_plus_rdkit_descriptors | Glass transition temperature (Tg, degC) | 245000 | 19000.6 |
| provided_plus_rdkit_descriptors | Viscosity (mPa.s) | 198495 | 19220.3 |
| provided_plus_rdkit_descriptors | rdkit_HallKierAlpha | 198371 | -29764.1 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState8 | 173900 | 11780.6 |
| provided_plus_rdkit_descriptors | rdkit_MolLogP | 134176 | -13087.1 |
| provided_plus_rdkit_descriptors | rdkit_AvgIpc | 114395 | -32035.8 |
| provided_plus_rdkit_descriptors | rdkit_MaxPartialCharge | 110543 | -31495.9 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA9 | 83996.3 | -1152.4 |
| provided_plus_rdkit_descriptors | rdkit_Chi0v | 77499.5 | 24191.7 |
| provided_plus_rdkit_descriptors | rdkit_MinPartialCharge | 72886.1 | -8352.28 |
| provided_plus_rdkit_descriptors | rdkit_MinEStateIndex | 72451.4 | 22116.5 |
| provided_plus_rdkit_descriptors | rdkit_PEOE_VSA11 | 70891.2 | -3513.29 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_MRHI | 68921.4 | 4914.19 |
| provided_plus_rdkit_descriptors | rdkit_MaxAbsEStateIndex | 68462.7 | -981.204 |
| provided_plus_rdkit_descriptors | rdkit_MinAbsEStateIndex | 62493.2 | 659.747 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_MWLOW | 57206.2 | -3289.1 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState2 | 56216.5 | -3456.21 |
| rdkit_descriptors_only | rdkit_VSA_EState3 | 654627 | 44372.6 |
| rdkit_descriptors_only | rdkit_EState_VSA10 | 646888 | -39301.2 |
| rdkit_descriptors_only | rdkit_Kappa2 | 541070 | 116144 |
| rdkit_descriptors_only | rdkit_VSA_EState7 | 524066 | 33228.2 |
| rdkit_descriptors_only | rdkit_HallKierAlpha | 466152 | -46447.6 |
| rdkit_descriptors_only | rdkit_BCUT2D_CHGLO | 420739 | -20772.8 |
| rdkit_descriptors_only | rdkit_BCUT2D_LOGPLOW | 315357 | -9156.28 |
| rdkit_descriptors_only | rdkit_BCUT2D_MRLOW | 306076 | -40874.5 |
| rdkit_descriptors_only | rdkit_MolLogP | 283946 | 1876.38 |
| rdkit_descriptors_only | rdkit_PEOE_VSA1 | 282820 | -46933.5 |
| rdkit_descriptors_only | rdkit_PEOE_VSA10 | 233482 | 16769.4 |
| rdkit_descriptors_only | rdkit_BCUT2D_CHGHI | 231787 | 66552 |
| rdkit_descriptors_only | rdkit_qed | 229334 | -18531.8 |
| rdkit_descriptors_only | rdkit_Kappa3 | 226768 | -27317 |
| rdkit_descriptors_only | rdkit_PEOE_VSA11 | 206992 | -3637.46 |
| rdkit_descriptors_only | rdkit_SMR_VSA5 | 173220 | 22538.1 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | 167972 | 2879.48 |
| rdkit_descriptors_only | rdkit_NumAtomStereoCenters | 135524 | 8821.37 |
| rdkit_descriptors_only | rdkit_Chi0 | 130725 | -35666.1 |
| rdkit_descriptors_only | rdkit_fr_Al_OH_noTert | 106658 | 21039.7 |

## SHAP Stability

| feature_set | feature | mean_abs_shap | cv_abs_shap | mean_rank | std_rank | presence_fraction | top_10_frequency | sign_consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| provided_only | Molecular weight (Da) | 965303 | 0.0828729 | 1.04 | 0.197949 | 1 | 1 | 0.56 |
| provided_only | Solubility (mg/mL) | 820797 | 0.0867465 | 1.96 | 0.197949 | 1 | 1 | 0.62 |
| provided_only | Glass transition temperature (Tg, degC) | 570088 | 0.101238 | 3 | 0 | 1 | 1 | 0.5 |
| provided_only | Viscosity (mPa.s) | 301327 | 0.175994 | 4.24 | 0.431419 | 1 | 1 | 0.54 |
| provided_only | Concentration (% w/v) | 251588 | 0.143463 | 4.76 | 0.431419 | 1 | 1 | 0.6 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA10 | 614076 | 0.216708 | 1.38 | 0.666701 | 1 | 1 | 0.54 |
| provided_plus_rdkit_descriptors | Solubility (mg/mL) | 539603 | 0.0979025 | 1.7 | 0.46291 | 1 | 1 | 0.5 |
| provided_plus_rdkit_descriptors | Concentration (% w/v) | 265074 | 0.120179 | 3.78 | 0.932191 | 1 | 1 | 0.6 |
| provided_plus_rdkit_descriptors | Glass transition temperature (Tg, degC) | 245000 | 0.166972 | 4.5 | 1.12938 | 1 | 1 | 0.68 |
| provided_plus_rdkit_descriptors | Viscosity (mPa.s) | 198495 | 0.180757 | 5.88 | 1.40901 | 1 | 1 | 0.74 |
| provided_plus_rdkit_descriptors | rdkit_HallKierAlpha | 198371 | 0.472838 | 7.78 | 7.64063 | 1 | 0.8 | 0.76 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState8 | 173900 | 0.266589 | 7.06 | 2.21691 | 1 | 0.94 | 0.66 |
| provided_plus_rdkit_descriptors | rdkit_MolLogP | 134176 | 0.532903 | 11.46 | 6.93471 | 1 | 0.58 | 0.64 |
| provided_plus_rdkit_descriptors | rdkit_AvgIpc | 114395 | 0.231417 | 10.44 | 2.83678 | 1 | 0.64 | 0.84 |
| provided_plus_rdkit_descriptors | rdkit_MaxPartialCharge | 110543 | 0.409681 | 11.88 | 5.01687 | 1 | 0.52 | 0.9 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA9 | 83996.3 | 0.506525 | 16.34 | 7.272 | 1 | 0.24 | 0.56 |
| provided_plus_rdkit_descriptors | rdkit_Chi0v | 77499.5 | 0.648432 | 18.92 | 10.5305 | 1 | 0.24 | 0.94 |
| provided_plus_rdkit_descriptors | rdkit_MinPartialCharge | 72886.1 | 0.422067 | 17.7 | 5.50046 | 1 | 0.14 | 0.76 |
| provided_plus_rdkit_descriptors | rdkit_MinEStateIndex | 72451.4 | 0.417613 | 17.72 | 7.03646 | 1 | 0.06 | 0.8 |
| provided_plus_rdkit_descriptors | rdkit_PEOE_VSA11 | 70891.2 | 0.434406 | 18.1 | 7.39346 | 1 | 0.14 | 0.6 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_MRHI | 68921.4 | 0.372907 | 17.64 | 5.95137 | 1 | 0.1 | 0.66 |
| provided_plus_rdkit_descriptors | rdkit_MaxAbsEStateIndex | 68462.7 | 0.259034 | 17.14 | 4.11076 | 1 | 0.02 | 0.5 |
| provided_plus_rdkit_descriptors | rdkit_MinAbsEStateIndex | 62493.2 | 0.405086 | 19.64 | 6.0702 | 1 | 0.06 | 0.56 |
| provided_plus_rdkit_descriptors | rdkit_BCUT2D_MWLOW | 57206.2 | 0.599562 | 22.92 | 8.90274 | 1 | 0.04 | 0.6 |
| provided_plus_rdkit_descriptors | rdkit_VSA_EState2 | 56216.5 | 0.542499 | 23.58 | 13.0651 | 1 | 0.04 | 0.56 |
| rdkit_descriptors_only | rdkit_VSA_EState3 | 654627 | 0.205549 | 2.10526 | 1.08527 | 0.76 | 1 | 0.684211 |
| rdkit_descriptors_only | rdkit_EState_VSA10 | 646888 | 0.197399 | 1.76 | 0.91607 | 1 | 1 | 0.56 |
| rdkit_descriptors_only | rdkit_Kappa2 | 541070 | nan | 4 | nan | 0.02 | 1 | 1 |
| rdkit_descriptors_only | rdkit_VSA_EState7 | 524066 | 0.253262 | 3.02 | 1.53184 | 1 | 1 | 0.62 |
| rdkit_descriptors_only | rdkit_HallKierAlpha | 466152 | 0.207753 | 3.42 | 1.38638 | 1 | 1 | 0.7 |
| rdkit_descriptors_only | rdkit_BCUT2D_CHGLO | 420739 | 0.155313 | 2.5 | 0.707107 | 0.04 | 1 | 0.5 |
| rdkit_descriptors_only | rdkit_BCUT2D_LOGPLOW | 315357 | 0.220503 | 5.92857 | 2.52291 | 0.56 | 0.928571 | 0.607143 |
| rdkit_descriptors_only | rdkit_BCUT2D_MRLOW | 306076 | nan | 5 | nan | 0.02 | 1 | 1 |
| rdkit_descriptors_only | rdkit_MolLogP | 283946 | 0.270783 | 6.72 | 2.07059 | 1 | 0.96 | 0.58 |
| rdkit_descriptors_only | rdkit_PEOE_VSA1 | 282820 | 0.248449 | 6.54 | 1.88669 | 1 | 0.98 | 0.82 |
| rdkit_descriptors_only | rdkit_PEOE_VSA10 | 233482 | 0.269445 | 8.04 | 2.39012 | 1 | 0.86 | 0.58 |
| rdkit_descriptors_only | rdkit_BCUT2D_CHGHI | 231787 | nan | 7 | nan | 0.02 | 1 | 1 |
| rdkit_descriptors_only | rdkit_qed | 229334 | 0.38022 | 8.53061 | 2.63867 | 0.98 | 0.816327 | 0.755102 |
| rdkit_descriptors_only | rdkit_Kappa3 | 226768 | 0.244261 | 8.29268 | 1.7211 | 0.82 | 0.902439 | 0.682927 |
| rdkit_descriptors_only | rdkit_PEOE_VSA11 | 206992 | 0.315318 | 9.18182 | 2.61199 | 0.44 | 0.681818 | 0.590909 |
| rdkit_descriptors_only | rdkit_SMR_VSA5 | 173220 | 0.302573 | 10.6667 | 2.33809 | 0.12 | 0.666667 | 0.666667 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | 167972 | 0.380175 | 10.2581 | 2.22063 | 0.62 | 0.548387 | 0.548387 |
| rdkit_descriptors_only | rdkit_NumAtomStereoCenters | 135524 | 0.203908 | 11.8929 | 1.79174 | 0.56 | 0.25 | 0.607143 |
| rdkit_descriptors_only | rdkit_Chi0 | 130725 | nan | 12 | nan | 0.02 | 0 | 1 |
| rdkit_descriptors_only | rdkit_fr_Al_OH_noTert | 106658 | 0.59733 | 13.25 | 3.19374 | 0.32 | 0.1875 | 0.9375 |

## SHAP Rank Stability By Feature Set

| feature_set | mean_spearman_rank_corr | std_spearman_rank_corr | mean_top10_jaccard | std_top10_jaccard | n_fold_pairs |
| --- | --- | --- | --- | --- | --- |
| provided_only | 0.954939 | 0.0571134 | 1 | 0 | 1225 |
| provided_plus_rdkit_descriptors | 0.902471 | 0.0206874 | 0.63614 | 0.126492 | 1225 |
| rdkit_descriptors_only | 0.69759 | 0.149883 | 0.662853 | 0.141607 | 1225 |

## High Feature Correlations

| feature_set | feature_1 | feature_2 | pearson_r | abs_pearson_r |
| --- | --- | --- | --- | --- |
| provided_plus_rdkit_descriptors | rdkit_fr_ketone | rdkit_fr_ketone_Topliss | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_fr_allylic_oxid | rdkit_fr_bicyclic | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_fr_Al_COO | rdkit_fr_COO2 | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_fr_Al_COO | rdkit_fr_COO | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_MaxAbsEStateIndex | rdkit_MaxEStateIndex | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_fr_COO | rdkit_fr_COO2 | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_SMR_VSA7 | rdkit_SlogP_VSA6 | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_SlogP_VSA4 | rdkit_EState_VSA6 | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_NumAliphaticRings | rdkit_RingCount | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA6 | rdkit_fr_bicyclic | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_EState_VSA6 | rdkit_fr_allylic_oxid | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_NumAmideBonds | rdkit_fr_amide | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_NumHeterocycles | rdkit_NumSaturatedHeterocycles | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_SMR_VSA3 | rdkit_fr_NH1 | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_NumAliphaticHeterocycles | rdkit_NumSaturatedHeterocycles | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_NumAliphaticHeterocycles | rdkit_NumHeterocycles | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_SlogP_VSA4 | rdkit_fr_allylic_oxid | 1 | 1 |
| provided_plus_rdkit_descriptors | rdkit_SlogP_VSA4 | rdkit_fr_bicyclic | 1 | 1 |
| provided_plus_rdkit_descriptors | Molecular weight (Da) | rdkit_MolWt | 1 | 1 |
| provided_plus_rdkit_descriptors | Molecular weight (Da) | rdkit_ExactMolWt | 1 | 1 |
| rdkit_descriptors_only | rdkit_SMR_VSA1 | rdkit_NOCount | 0.99948 | 0.99948 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | rdkit_NumHAcceptors | 0.999309 | 0.999309 |
| rdkit_descriptors_only | rdkit_NOCount | rdkit_NumHeteroatoms | 0.998635 | 0.998635 |
| rdkit_descriptors_only | rdkit_TPSA | rdkit_NOCount | 0.998494 | 0.998494 |
| rdkit_descriptors_only | rdkit_SMR_VSA1 | rdkit_NumHAcceptors | 0.998481 | 0.998481 |
| rdkit_descriptors_only | rdkit_NOCount | rdkit_NumHAcceptors | 0.998472 | 0.998472 |
| rdkit_descriptors_only | rdkit_SMR_VSA1 | rdkit_TPSA | 0.998435 | 0.998435 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | rdkit_NOCount | 0.997783 | 0.997783 |
| rdkit_descriptors_only | rdkit_SMR_VSA1 | rdkit_SlogP_VSA2 | 0.997687 | 0.997687 |
| rdkit_descriptors_only | rdkit_SMR_VSA1 | rdkit_NumHeteroatoms | 0.99746 | 0.99746 |
| rdkit_descriptors_only | rdkit_VSA_EState3 | rdkit_NumHDonors | 0.9973 | 0.9973 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | rdkit_NumHeteroatoms | 0.997104 | 0.997104 |
| rdkit_descriptors_only | rdkit_NumHAcceptors | rdkit_NumHeteroatoms | 0.996837 | 0.996837 |
| rdkit_descriptors_only | rdkit_VSA_EState3 | rdkit_NumHAcceptors | 0.996158 | 0.996158 |
| rdkit_descriptors_only | rdkit_TPSA | rdkit_NumHAcceptors | 0.996068 | 0.996068 |
| rdkit_descriptors_only | rdkit_PEOE_VSA1 | rdkit_PEOE_VSA10 | 0.99605 | 0.99605 |
| rdkit_descriptors_only | rdkit_TPSA | rdkit_NumHeteroatoms | 0.995973 | 0.995973 |
| rdkit_descriptors_only | rdkit_TPSA | rdkit_NHOHCount | 0.995877 | 0.995877 |
| rdkit_descriptors_only | rdkit_SlogP_VSA2 | rdkit_VSA_EState3 | 0.995866 | 0.995866 |
| rdkit_descriptors_only | rdkit_NHOHCount | rdkit_NumHDonors | 0.995621 | 0.995621 |

## XGBoost All-Feature Values vs Target Correlations

- Full Pearson matrix: `results\feature_analysis\xgboost_all_feature_pearson_matrix.csv`
- Full Spearman matrix: `results\feature_analysis\xgboost_all_feature_spearman_matrix.csv`

| feature | used_by_xgboost_after_pipeline | pearson_r_with_target | spearman_r_with_target | abs_pearson_r_with_target |
| --- | --- | --- | --- | --- |
| rdkit_HallKierAlpha | True | 0.42158 | 0.409062 | 0.42158 |
| Solubility (mg/mL) | True | 0.395276 | 0.145041 | 0.395276 |
| rdkit_EState_VSA10 | True | -0.388469 | -0.515028 | 0.388469 |
| rdkit_VSA_EState7 | True | 0.37846 | 0.483656 | 0.37846 |
| rdkit_MolLogP | True | 0.369255 | 0.462648 | 0.369255 |
| rdkit_qed | True | 0.354588 | 0.435485 | 0.354588 |
| rdkit_SlogP_VSA3 | True | -0.353166 | -0.486248 | 0.353166 |
| rdkit_PEOE_VSA1 | True | -0.352591 | -0.456139 | 0.352591 |
| rdkit_TPSA | True | -0.348044 | -0.42859 | 0.348044 |
| rdkit_NOCount | True | -0.347193 | -0.449405 | 0.347193 |
| rdkit_PEOE_VSA10 | True | -0.34512 | -0.38117 | 0.34512 |
| rdkit_SMR_VSA1 | True | -0.345068 | -0.42081 | 0.345068 |
| rdkit_NumHDonors | True | -0.344299 | -0.421401 | 0.344299 |
| rdkit_BCUT2D_LOGPLOW | True | 0.343764 | 0.443595 | 0.343764 |
| rdkit_NumHAcceptors | True | -0.339739 | -0.436653 | 0.339739 |
| rdkit_NumHeteroatoms | True | -0.337304 | -0.414218 | 0.337304 |
| rdkit_NHOHCount | True | -0.33612 | -0.422294 | 0.33612 |
| rdkit_Kappa3 | True | -0.335949 | -0.403073 | 0.335949 |
| rdkit_VSA_EState3 | True | -0.331665 | -0.387249 | 0.331665 |
| rdkit_SlogP_VSA2 | True | -0.330755 | -0.412812 | 0.330755 |
| rdkit_NumAtomStereoCenters | True | -0.330727 | -0.454576 | 0.330727 |
| rdkit_EState_VSA1 | True | -0.329739 | -0.374676 | 0.329739 |
| rdkit_fr_Al_OH_noTert | True | -0.328511 | -0.361191 | 0.328511 |
| rdkit_PEOE_VSA11 | True | -0.328496 | -0.444894 | 0.328496 |
| rdkit_fr_Al_OH | True | -0.3267 | -0.359404 | 0.3267 |
| rdkit_fr_ether | True | -0.324404 | -0.411709 | 0.324404 |
| rdkit_Kappa2 | True | -0.323019 | -0.366499 | 0.323019 |
| rdkit_Chi0 | True | -0.320003 | -0.438925 | 0.320003 |
| rdkit_Chi1 | True | -0.318997 | -0.427401 | 0.318997 |
| rdkit_NumValenceElectrons | True | -0.318214 | -0.426858 | 0.318214 |

## Output Files

- `results\feature_analysis\fold_metrics.csv`: metrics
- `results\feature_analysis\cv_predictions.csv`: predictions
- `results\feature_analysis\cv_train_predictions.csv`: train_predictions
- `results\feature_analysis\feature_importance_fold.csv`: importance_fold
- `results\feature_analysis\feature_importance_summary.csv`: importance_summary
- `results\feature_analysis\shap_fold_feature_summary.csv`: shap_fold
- `results\feature_analysis\shap_summary.csv`: shap_summary
- `results\feature_analysis\shap_stability_summary.csv`: shap_stability
- `results\feature_analysis\shap_rank_stability_fold_pairs.csv`: shap_rank_stability_pairs
- `results\feature_analysis\shap_rank_stability_summary.csv`: shap_rank_stability_summary
- `results\feature_analysis\feature_correlation_summary.csv`: correlation_summary
- `results\feature_analysis\feature_correlation_high_pairs.csv`: correlation_pairs
- `results\feature_analysis\feature_correlation_matrices`: correlation_matrix_dir
- `results\feature_analysis\xgboost_all_feature_values_wide.csv`: xgboost_all_feature_values_wide
- `results\feature_analysis\xgboost_all_feature_values_long.csv`: xgboost_all_feature_values_long
- `results\feature_analysis\xgboost_all_feature_target_correlations.csv`: xgboost_all_feature_target_correlations
- `results\feature_analysis\xgboost_all_feature_pearson_matrix.csv`: xgboost_all_feature_pearson_matrix
- `results\feature_analysis\xgboost_all_feature_spearman_matrix.csv`: xgboost_all_feature_spearman_matrix
- `results\feature_analysis\data_profile.json`: profile
