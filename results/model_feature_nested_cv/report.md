# Model and feature nested CV evaluation

- Data: `dataset.csv`
- Samples: 300
- Repeated CV: 10 x 5-fold
- Inner tuning: 3-fold RandomizedSearchCV in every outer fold, n_iter=80
- Parameter profile: `anti_overfit`
- Hyperparameter selection rule: `one_standard_error`
- One-SE multiplier: 1.0
- Feature selection: `auto`
- Selected model for downstream analyses: `XGBoost`
- Best scenario: `XGBoost__provided_plus_rdkit_descriptors`
- Best mean fold RMSE: 962549.909
- Best mean fold R2: 0.838782
- Metric wide tables: `results\model_feature_nested_cv\metric_wide\repeat_mean`

## CV Summary

| scenario_run | model | feature_set | mean_fold_rmse | std_fold_rmse | mean_fold_mae | mean_fold_r2 | std_fold_r2 | mean_train_rmse | mean_train_mae | mean_train_r2 | pooled_train_rmse | pooled_train_r2 | mean_train_val_rmse_gap | mean_train_val_rmse_ratio | mean_train_val_r2_gap | mean_inner_selected_rmse_gap_vs_min | mean_one_se_candidate_count | pooled_oof_rmse | pooled_oof_r2 | pooled_oof_pearson | pooled_oof_spearman | n_predictions | consensus_best_params_count | consensus_best_params_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost__provided_plus_rdkit_descriptors | XGBoost | provided_plus_rdkit_descriptors | 962550 | 183688 | 675925 | 0.838782 | 0.0576966 | 418068 | 291443 | 0.96205 | 481076 | 0.961888 | 544482 | 3.23203 | 0.123267 | 46283.9 | 10.26 | 979576 | 0.841982 | 0.917609 | 0.942611 | 3000 | 2 | 0.04 |
| GBoost__provided_plus_rdkit_descriptors | GBoost | provided_plus_rdkit_descriptors | 1.00461e+06 | 197207 | 705648 | 0.825429 | 0.0589932 | 524238 | 369994 | 0.945469 | 573398 | 0.945857 | 480372 | 2.79464 | 0.120039 | 49026 | 14.02 | 1.0234e+06 | 0.827526 | 0.909693 | 0.938421 | 3000 | 3 | 0.06 |
| KNN__provided_plus_rdkit_descriptors | KNN | provided_plus_rdkit_descriptors | 1.07968e+06 | 152646 | 753707 | 0.799784 | 0.053466 | 752740 | 511502 | 0.902643 | 768705 | 0.902692 | 326937 | 4.94281 | 0.102859 | 26354.9 | 12.08 | 1.0902e+06 | 0.804276 | 0.898043 | 0.938587 | 3000 | 18 | 0.36 |
| GBoost__provided_only | GBoost | provided_only | 1.11496e+06 | 166203 | 824004 | 0.786962 | 0.0542009 | 367048 | 270462 | 0.973819 | 398124 | 0.973898 | 747913 | 3.92226 | 0.186857 | 56272.2 | 13.94 | 1.12704e+06 | 0.790826 | 0.890472 | 0.895891 | 3000 | 3 | 0.06 |
| XGBoost__provided_only | XGBoost | provided_only | 1.12571e+06 | 153605 | 830367 | 0.783096 | 0.0520485 | 389009 | 283095 | 0.970474 | 424807 | 0.970282 | 736701 | 3.63043 | 0.187378 | 63557.6 | 11.04 | 1.13593e+06 | 0.78751 | 0.889713 | 0.893523 | 3000 | 3 | 0.06 |
| RandomForest__provided_plus_rdkit_descriptors | RandomForest | provided_plus_rdkit_descriptors | 1.32291e+06 | 179410 | 972392 | 0.701935 | 0.0671619 | 1.06352e+06 | 772850 | 0.812368 | 1.06665e+06 | 0.81264 | 259388 | 1.25087 | 0.110433 | 32274.3 | 3.84 | 1.33477e+06 | 0.706609 | 0.848294 | 0.88183 | 3000 | 5 | 0.1 |
| KNN__rdkit_descriptors_only | KNN | rdkit_descriptors_only | 1.37374e+06 | 160681 | 940220 | 0.67885 | 0.063283 | 1.15857e+06 | 756580 | 0.778395 | 1.15943e+06 | 0.778629 | 215168 | 1.19249 | 0.0995452 | 33400.3 | 20.14 | 1.38292e+06 | 0.685063 | 0.829454 | 0.875867 | 3000 | 24 | 0.48 |
| XGBoost__rdkit_descriptors_only | XGBoost | rdkit_descriptors_only | 1.40878e+06 | 165403 | 992218 | 0.6631 | 0.0624488 | 1.19259e+06 | 811386 | 0.76506 | 1.19362e+06 | 0.765382 | 216194 | 1.18657 | 0.101959 | 48111.8 | 37.92 | 1.41827e+06 | 0.668757 | 0.817826 | 0.859421 | 3000 | 3 | 0.06 |
| GBoost__rdkit_descriptors_only | GBoost | rdkit_descriptors_only | 1.41472e+06 | 182879 | 992886 | 0.660054 | 0.0689074 | 1.19054e+06 | 807657 | 0.765659 | 1.19167e+06 | 0.766145 | 224177 | 1.19242 | 0.105606 | 43318.2 | 39.88 | 1.42626e+06 | 0.665015 | 0.815653 | 0.859212 | 3000 | 3 | 0.06 |
| RandomForest__rdkit_descriptors_only | RandomForest | rdkit_descriptors_only | 1.54131e+06 | 178349 | 1.13203e+06 | 0.597019 | 0.0745053 | 1.31556e+06 | 946183 | 0.713183 | 1.31825e+06 | 0.713827 | 225749 | 1.17499 | 0.116164 | 45716.1 | 7.94 | 1.55139e+06 | 0.603657 | 0.780055 | 0.824751 | 3000 | 4 | 0.08 |
| DecisionTree__provided_plus_rdkit_descriptors | DecisionTree | provided_plus_rdkit_descriptors | 1.55428e+06 | 238105 | 1.13728e+06 | 0.585173 | 0.113921 | 1.28301e+06 | 909744 | 0.72539 | 1.28993e+06 | 0.725993 | 271268 | 1.21944 | 0.140217 | 43142.4 | 5.26 | 1.57205e+06 | 0.593027 | 0.771514 | 0.799863 | 3000 | 2 | 0.04 |
| RandomForest__provided_only | RandomForest | provided_only | 1.60308e+06 | 197894 | 1.22259e+06 | 0.563836 | 0.0842207 | 1.29587e+06 | 972683 | 0.720329 | 1.30174e+06 | 0.72095 | 307203 | 1.24674 | 0.156492 | 54400.3 | 6.1 | 1.615e+06 | 0.570486 | 0.766974 | 0.786782 | 3000 | 4 | 0.08 |
| DecisionTree__rdkit_descriptors_only | DecisionTree | rdkit_descriptors_only | 1.65538e+06 | 183750 | 1.20563e+06 | 0.534879 | 0.08314 | 1.41653e+06 | 1.0022e+06 | 0.668014 | 1.41854e+06 | 0.668629 | 238848 | 1.17282 | 0.133135 | 32414.7 | 4.82 | 1.66534e+06 | 0.543293 | 0.739667 | 0.778731 | 3000 | 2 | 0.04 |
| DecisionTree__provided_only | DecisionTree | provided_only | 1.89999e+06 | 276718 | 1.42738e+06 | 0.374636 | 0.189291 | 1.58554e+06 | 1.17034e+06 | 0.578175 | 1.59794e+06 | 0.579512 | 314452 | 1.20836 | 0.203539 | 50578.5 | 12.14 | 1.91963e+06 | 0.393168 | 0.635636 | 0.638511 | 3000 | 2 | 0.04 |
| KNN__provided_only | KNN | provided_only | 2.11825e+06 | 189591 | 1.71035e+06 | 0.242515 | 0.087584 | 549499 | 450813 | 0.807554 | 1.07832e+06 | 0.80852 | 1.56875e+06 | 1.04037 | 0.565038 | 57861 | 7.86 | 2.12655e+06 | 0.255297 | 0.522131 | 0.503511 | 3000 | 10 | 0.2 |

## Best RMSE vs 1-SE Selection Rule Comparison

| selection_rule | scenario_run | model | feature_set | mean_fold_rmse | std_fold_rmse | mean_fold_mae | mean_fold_r2 | std_fold_r2 | mean_train_rmse | mean_train_r2 | pooled_train_rmse | pooled_train_r2 | mean_train_val_rmse_gap | mean_train_val_rmse_ratio | mean_train_val_r2_gap | mean_inner_selected_rmse_gap_vs_min | mean_one_se_candidate_count | pooled_oof_rmse | pooled_oof_r2 | pooled_oof_pearson | pooled_oof_spearman | n_predictions | consensus_best_params_count | consensus_best_params_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_rmse | XGBoost__provided_plus_rdkit_descriptors | XGBoost | provided_plus_rdkit_descriptors | 883856 | 165066 | 607530 | 0.865132 | 0.0452313 | 204233 | 0.991066 | 232831 | 0.991073 | 679623 | 6.44896 | 0.125934 | 0 | 1 | 898834 | 0.866958 | 0.931276 | 0.952332 | 3000 | 4 | 0.08 |
| best_rmse | GBoost__provided_plus_rdkit_descriptors | GBoost | provided_plus_rdkit_descriptors | 928485 | 170965 | 640882 | 0.850989 | 0.0499168 | 286199 | 0.982016 | 329886 | 0.982079 | 642286 | 9.16085 | 0.131027 | 0 | 1 | 943784 | 0.853318 | 0.923791 | 0.947004 | 3000 | 4 | 0.08 |
| best_rmse | XGBoost__provided_only | XGBoost | provided_only | 1.05501e+06 | 157227 | 773303 | 0.809442 | 0.0491862 | 213064 | 0.99004 | 247096 | 0.989945 | 841942 | 6.90957 | 0.180598 | 0 | 1 | 1.06642e+06 | 0.81272 | 0.903948 | 0.908989 | 3000 | 3 | 0.06 |
| best_rmse | GBoost__provided_only | GBoost | provided_only | 1.0731e+06 | 162153 | 779182 | 0.802855 | 0.0505964 | 219615 | 0.989105 | 255573 | 0.989244 | 853482 | 7.64498 | 0.18625 | 0 | 1 | 1.08504e+06 | 0.806126 | 0.898913 | 0.907132 | 3000 | 3 | 0.06 |
| best_rmse | KNN__provided_plus_rdkit_descriptors | KNN | provided_plus_rdkit_descriptors | 1.09803e+06 | 161433 | 763093 | 0.792109 | 0.0597394 | 677572 | 0.911841 | 730902 | 0.912027 | 420457 | 8.64455 | 0.119732 | 0 | 1 | 1.1096e+06 | 0.797249 | 0.894193 | 0.93514 | 3000 | 5 | 0.1 |
| best_rmse | RandomForest__provided_plus_rdkit_descriptors | RandomForest | provided_plus_rdkit_descriptors | 1.2658e+06 | 170333 | 927714 | 0.72791 | 0.056619 | 1.00709e+06 | 0.832096 | 1.00915e+06 | 0.832297 | 258711 | 1.26432 | 0.104186 | 0 | 1 | 1.27698e+06 | 0.731464 | 0.862111 | 0.894382 | 3000 | 5 | 0.1 |
| best_rmse | KNN__rdkit_descriptors_only | KNN | rdkit_descriptors_only | 1.35821e+06 | 162688 | 929674 | 0.685649 | 0.0646823 | 1.16366e+06 | 0.776373 | 1.16457e+06 | 0.776661 | 194553 | 1.17377 | 0.0907236 | 0 | 1 | 1.36773e+06 | 0.691944 | 0.833343 | 0.877017 | 3000 | 6 | 0.12 |
| best_rmse | XGBoost__rdkit_descriptors_only | XGBoost | rdkit_descriptors_only | 1.37168e+06 | 166683 | 944344 | 0.679896 | 0.0637418 | 1.16392e+06 | 0.776366 | 1.16458e+06 | 0.776658 | 207763 | 1.18417 | 0.0964697 | 0 | 1 | 1.38157e+06 | 0.685675 | 0.828829 | 0.871667 | 3000 | 3 | 0.06 |
| best_rmse | GBoost__rdkit_descriptors_only | GBoost | rdkit_descriptors_only | 1.37221e+06 | 165744 | 947233 | 0.679768 | 0.0634712 | 1.16482e+06 | 0.776062 | 1.16545e+06 | 0.776323 | 207393 | 1.18378 | 0.096294 | 0 | 1 | 1.38198e+06 | 0.685488 | 0.828633 | 0.871096 | 3000 | 2 | 0.04 |
| best_rmse | RandomForest__rdkit_descriptors_only | RandomForest | rdkit_descriptors_only | 1.47839e+06 | 148998 | 1.07343e+06 | 0.629253 | 0.0621212 | 1.25755e+06 | 0.738948 | 1.25824e+06 | 0.73929 | 220847 | 1.17966 | 0.109695 | 0 | 1 | 1.48573e+06 | 0.636493 | 0.799777 | 0.841433 | 3000 | 5 | 0.1 |
| best_rmse | RandomForest__provided_only | RandomForest | provided_only | 1.53249e+06 | 189143 | 1.16444e+06 | 0.601247 | 0.0780768 | 1.19648e+06 | 0.763171 | 1.19844e+06 | 0.763483 | 336003 | 1.28799 | 0.161924 | 0 | 1 | 1.54388e+06 | 0.607482 | 0.79087 | 0.810273 | 3000 | 5 | 0.1 |
| best_rmse | DecisionTree__provided_plus_rdkit_descriptors | DecisionTree | provided_plus_rdkit_descriptors | 1.5351e+06 | 269812 | 1.11267e+06 | 0.598448 | 0.110873 | 1.23267e+06 | 0.746334 | 1.2395e+06 | 0.746997 | 302427 | 1.25324 | 0.147886 | 0 | 1 | 1.55816e+06 | 0.600186 | 0.777361 | 0.819692 | 3000 | 3 | 0.06 |
| best_rmse | DecisionTree__rdkit_descriptors_only | DecisionTree | rdkit_descriptors_only | 1.63351e+06 | 179953 | 1.17848e+06 | 0.544952 | 0.0929498 | 1.39188e+06 | 0.679421 | 1.39416e+06 | 0.679923 | 241633 | 1.17948 | 0.134469 | 0 | 1 | 1.6432e+06 | 0.555358 | 0.748437 | 0.791478 | 3000 | 6 | 0.12 |
| best_rmse | DecisionTree__provided_only | DecisionTree | provided_only | 1.82892e+06 | 224228 | 1.35577e+06 | 0.42593 | 0.1422 | 1.46624e+06 | 0.641923 | 1.47275e+06 | 0.642818 | 362684 | 1.25383 | 0.215993 | 0 | 1 | 1.84235e+06 | 0.44105 | 0.672659 | 0.672189 | 3000 | 5 | 0.1 |
| best_rmse | KNN__provided_only | KNN | provided_only | 2.03514e+06 | 214100 | 1.58743e+06 | 0.298362 | 0.115273 | 0 | 1 | 0 | 1 | 2.03514e+06 | nan | 0.701638 | 0 | 1 | 2.04615e+06 | 0.310546 | 0.558224 | 0.531167 | 3000 | 33 | 0.66 |
| one_standard_error | XGBoost__provided_plus_rdkit_descriptors | XGBoost | provided_plus_rdkit_descriptors | 962550 | 183688 | 675925 | 0.838782 | 0.0576966 | 418068 | 0.96205 | 481076 | 0.961888 | 544482 | 3.23203 | 0.123267 | 46283.9 | 10.26 | 979576 | 0.841982 | 0.917609 | 0.942611 | 3000 | 2 | 0.04 |
| one_standard_error | GBoost__provided_plus_rdkit_descriptors | GBoost | provided_plus_rdkit_descriptors | 1.00461e+06 | 197207 | 705648 | 0.825429 | 0.0589932 | 524238 | 0.945469 | 573398 | 0.945857 | 480372 | 2.79464 | 0.120039 | 49026 | 14.02 | 1.0234e+06 | 0.827526 | 0.909693 | 0.938421 | 3000 | 3 | 0.06 |
| one_standard_error | KNN__provided_plus_rdkit_descriptors | KNN | provided_plus_rdkit_descriptors | 1.07968e+06 | 152646 | 753707 | 0.799784 | 0.053466 | 752740 | 0.902643 | 768705 | 0.902692 | 326937 | 4.94281 | 0.102859 | 26354.9 | 12.08 | 1.0902e+06 | 0.804276 | 0.898043 | 0.938587 | 3000 | 18 | 0.36 |
| one_standard_error | GBoost__provided_only | GBoost | provided_only | 1.11496e+06 | 166203 | 824004 | 0.786962 | 0.0542009 | 367048 | 0.973819 | 398124 | 0.973898 | 747913 | 3.92226 | 0.186857 | 56272.2 | 13.94 | 1.12704e+06 | 0.790826 | 0.890472 | 0.895891 | 3000 | 3 | 0.06 |
| one_standard_error | XGBoost__provided_only | XGBoost | provided_only | 1.12571e+06 | 153605 | 830367 | 0.783096 | 0.0520485 | 389009 | 0.970474 | 424807 | 0.970282 | 736701 | 3.63043 | 0.187378 | 63557.6 | 11.04 | 1.13593e+06 | 0.78751 | 0.889713 | 0.893523 | 3000 | 3 | 0.06 |
| one_standard_error | RandomForest__provided_plus_rdkit_descriptors | RandomForest | provided_plus_rdkit_descriptors | 1.32291e+06 | 179410 | 972392 | 0.701935 | 0.0671619 | 1.06352e+06 | 0.812368 | 1.06665e+06 | 0.81264 | 259388 | 1.25087 | 0.110433 | 32274.3 | 3.84 | 1.33477e+06 | 0.706609 | 0.848294 | 0.88183 | 3000 | 5 | 0.1 |
| one_standard_error | KNN__rdkit_descriptors_only | KNN | rdkit_descriptors_only | 1.37374e+06 | 160681 | 940220 | 0.67885 | 0.063283 | 1.15857e+06 | 0.778395 | 1.15943e+06 | 0.778629 | 215168 | 1.19249 | 0.0995452 | 33400.3 | 20.14 | 1.38292e+06 | 0.685063 | 0.829454 | 0.875867 | 3000 | 24 | 0.48 |
| one_standard_error | XGBoost__rdkit_descriptors_only | XGBoost | rdkit_descriptors_only | 1.40878e+06 | 165403 | 992218 | 0.6631 | 0.0624488 | 1.19259e+06 | 0.76506 | 1.19362e+06 | 0.765382 | 216194 | 1.18657 | 0.101959 | 48111.8 | 37.92 | 1.41827e+06 | 0.668757 | 0.817826 | 0.859421 | 3000 | 3 | 0.06 |
| one_standard_error | GBoost__rdkit_descriptors_only | GBoost | rdkit_descriptors_only | 1.41472e+06 | 182879 | 992886 | 0.660054 | 0.0689074 | 1.19054e+06 | 0.765659 | 1.19167e+06 | 0.766145 | 224177 | 1.19242 | 0.105606 | 43318.2 | 39.88 | 1.42626e+06 | 0.665015 | 0.815653 | 0.859212 | 3000 | 3 | 0.06 |
| one_standard_error | RandomForest__rdkit_descriptors_only | RandomForest | rdkit_descriptors_only | 1.54131e+06 | 178349 | 1.13203e+06 | 0.597019 | 0.0745053 | 1.31556e+06 | 0.713183 | 1.31825e+06 | 0.713827 | 225749 | 1.17499 | 0.116164 | 45716.1 | 7.94 | 1.55139e+06 | 0.603657 | 0.780055 | 0.824751 | 3000 | 4 | 0.08 |
| one_standard_error | DecisionTree__provided_plus_rdkit_descriptors | DecisionTree | provided_plus_rdkit_descriptors | 1.55428e+06 | 238105 | 1.13728e+06 | 0.585173 | 0.113921 | 1.28301e+06 | 0.72539 | 1.28993e+06 | 0.725993 | 271268 | 1.21944 | 0.140217 | 43142.4 | 5.26 | 1.57205e+06 | 0.593027 | 0.771514 | 0.799863 | 3000 | 2 | 0.04 |
| one_standard_error | RandomForest__provided_only | RandomForest | provided_only | 1.60308e+06 | 197894 | 1.22259e+06 | 0.563836 | 0.0842207 | 1.29587e+06 | 0.720329 | 1.30174e+06 | 0.72095 | 307203 | 1.24674 | 0.156492 | 54400.3 | 6.1 | 1.615e+06 | 0.570486 | 0.766974 | 0.786782 | 3000 | 4 | 0.08 |
| one_standard_error | DecisionTree__rdkit_descriptors_only | DecisionTree | rdkit_descriptors_only | 1.65538e+06 | 183750 | 1.20563e+06 | 0.534879 | 0.08314 | 1.41653e+06 | 0.668014 | 1.41854e+06 | 0.668629 | 238848 | 1.17282 | 0.133135 | 32414.7 | 4.82 | 1.66534e+06 | 0.543293 | 0.739667 | 0.778731 | 3000 | 2 | 0.04 |
| one_standard_error | DecisionTree__provided_only | DecisionTree | provided_only | 1.89999e+06 | 276718 | 1.42738e+06 | 0.374636 | 0.189291 | 1.58554e+06 | 0.578175 | 1.59794e+06 | 0.579512 | 314452 | 1.20836 | 0.203539 | 50578.5 | 12.14 | 1.91963e+06 | 0.393168 | 0.635636 | 0.638511 | 3000 | 2 | 0.04 |
| one_standard_error | KNN__provided_only | KNN | provided_only | 2.11825e+06 | 189591 | 1.71035e+06 | 0.242515 | 0.087584 | 549499 | 0.807554 | 1.07832e+06 | 0.80852 | 1.56875e+06 | 1.04037 | 0.565038 | 57861 | 7.86 | 2.12655e+06 | 0.255297 | 0.522131 | 0.503511 | 3000 | 10 | 0.2 |

## Output Files

- `results\model_feature_nested_cv\cv_summary.csv`: summary
- `results\model_feature_nested_cv\cv_summary_selection_rules.csv`: selection_summary
- `results\model_feature_nested_cv\repeat_metrics.csv`: repeat_metrics
- `results\model_feature_nested_cv\repeat_metrics_selection_rules.csv`: selection_repeat_metrics
- `results\model_feature_nested_cv\fold_metrics.csv`: fold_metrics
- `results\model_feature_nested_cv\fold_metrics_selection_rules.csv`: selection_fold_metrics
- `results\model_feature_nested_cv\cv_predictions.csv`: predictions
- `results\model_feature_nested_cv\cv_predictions_selection_rules.csv`: selection_predictions
- `results\model_feature_nested_cv\cv_train_predictions.csv`: train_predictions
- `results\model_feature_nested_cv\cv_train_predictions_selection_rules.csv`: selection_train_predictions
- `results\model_feature_nested_cv\best_hyperparameters.csv`: best_params
- `results\model_feature_nested_cv\best_hyperparameters_selection_rules.csv`: selection_best_params
- `results\model_feature_nested_cv\selected_model.json`: selected_model
- `results\model_feature_nested_cv\metric_wide\repeat_mean`: metric_wide_dir
- `results\model_feature_nested_cv\smiles_cleaning_report.csv`: cleaning
- `results\model_feature_nested_cv\data_profile.json`: profile
