from fns import build_feature_table, filter_test_data, filter_train_data, data_preprocess
from featurevectors import StaticRadiomics, MultiphaseRadiomics, EngineeredTempRadiomics, TimeSeriesRadiomics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pysera


def main():
    #datapath = Path("../breast1/NACT_extracted")
    datapath = Path("../MAMA_MIA_ISPY2/images")
    clinpath = Path("../clinical_and_imaging_info.xlsx")
    comb = "S"
    preprocess = False
    #print(dataset["NACT_01"]["time_res"])
    # grid search of C for tuning

    c_vals = [0.001, 0.01, 0.1, 1]
    ks = [5, 10, 25, 50]


    x, y, feature_names, patient_ids = build_feature_table(dataset_name="ispy2", num_phases=6, datapath=datapath, clinpath=clinpath, comb=comb, preprocess=preprocess, comp=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lst_accu_stratified = []
    aucs = []

    for train_index, test_index in skf.split(x, y):
        model = LogisticRegression(penalty='l1', random_state=42, solver='saga', max_iter=5000, class_weight='balanced') # max_iter is a ceiling.
        x_train_fold, x_test_fold = x[train_index], x[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        x_train_fold, feature_names_filtered, vf, retain, selector, scalerobject = filter_train_data(x_train_fold, y_train_fold, feature_names=feature_names, k=25)
        x_test_fold = filter_test_data(x_test_fold, vf_object=vf, retain_matrix=retain, selector_object=selector, scaler_object=scalerobject)

        print("selected features: ", feature_names_filtered)

        model.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(model.score(x_test_fold, y_test_fold))

        y_probability = model.predict_proba(x_test_fold)[:, 1]
        fold_auc = roc_auc_score(y_test_fold, y_probability)
        aucs.append(fold_auc)
        print("Classes:", model.classes_)
        print("True labels:", np.unique(y_test_fold, return_counts=True))
        print("Predicted labels:", np.unique(model.predict(x_test_fold), return_counts=True))

    print("Fold accuracies:", lst_accu_stratified)
    print("Mean accuracy:", np.mean(lst_accu_stratified))
    print("Fold ROC-AUCs:", aucs)
    print("Mean ROC-AUC:", np.mean(aucs))

    # checking input
    print(x.shape)
    print(np.unique(y, return_counts=True))

if __name__ == "__main__":
    main()
