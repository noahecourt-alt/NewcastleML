# This script is for data preparation. It includes the entire feature table creation, as well as the feature selection pipeline.

import numpy as np
from collections import defaultdict
from radiomics import featureextractor
import SimpleITK as sitk
from pathlib import Path
from typing import Dict
from featurevectors import StaticRadiomics, MultiphaseRadiomics, TimeSeriesRadiomics, EngineeredTempRadiomics
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast

# loop through the patients. needs to extract the defined features. needs to return the relevant patient feature_table X and corresponding outcome_table Y. 
# Needs to return X, Y, feature_names, patient number

def create_feature_table(dataset: Dict, comb: str):
    # loop through dataset
    comb_list = comb.split('_')
    feature_objects = []
    extractor = featureextractor.RadiomicsFeatureExtractor()
    statrad = StaticRadiomics(extractor)
    multirad = MultiphaseRadiomics(statrad)
    timeseriesrad = TimeSeriesRadiomics(multirad)
    engineeredrad = EngineeredTempRadiomics(multirad)

    if "S" in comb_list:
        feature_objects.append(statrad)
    if "M" in comb_list:
        feature_objects.append(multirad)
        if "S" in comb_list:
            feature_objects.remove(statrad)
    if "E" in comb_list:
        feature_objects.append(engineeredrad)
    if "T" in comb_list:
        feature_objects.append(timeseriesrad)
    x = []
    y = []
    feature_names = None
    patient_ids = []

    for patient_id, patient in dataset.items():
        if "M" in comb_list or "E" in comb_list or "T" in comb_list: # since E and T depend on M extracting (for the time series table)
            multirad.extract(patient["images"], patient["mask"])
        if "S" in comb_list and "M" not in comb_list: #extracts the single phase radiomics
            statrad.extract(patient["images"]["1"], patient["mask"]) #patient["phase1"] as placeholder logic for grabbing the decided phase. 
        if "E" in comb_list:
            engineeredrad.apply_time_res(patient["time_res"])

        patient_vector = []
        total_feature_names = []
        for feature_object in feature_objects:
            vector, names = feature_object.get_feature_vector()
            patient_vector.extend(vector)
            total_feature_names.extend(names)

        if feature_names is None:   
            feature_names = total_feature_names
        else:
            assert feature_names == total_feature_names, "There are missing features"

        x.append(patient_vector) #patient_vector is a row, so this results in 2D array as wanted. 
        y.append(patient["pcr"])
        patient_ids.append(patient_id)
    x = np.array(x, dtype=np.float32)
    y = np.array(y)

    return x, y, feature_names, patient_ids

def variancefilter(x, feature_names, threshold: float=0.0): # helper function to use sklearn variance filter
    vf = VarianceThreshold(threshold=threshold) # transforms across the columns, as we would want (variance along the same feature across the patients)
    x_filtered = vf.fit_transform(x)
    feature_names_filtered = np.array(feature_names)[vf.get_support()]
    return x_filtered, feature_names_filtered, vf

def pearsonfilter(x, feature_names, threshold: float=0.7): # 0.3 base taken from Huang et al. review, since a threshold of 0.3 is harsh and will remove many features. 0.85 for Kovacevic et al. prob meant 0.7 from huang et al
    # remove highly correlated features 
    corr_matrix = np.corrcoef(x, rowvar=False) # output symmetric corr matrix. features x features
    retain = np.ones(x.shape[1], dtype=bool) # [True, True, True, etc.] of len(x[0])

    for i in range(x.shape[1]):
        if not retain[i]: # since it is already removed. 
            continue
        for j in range(i+1, x.shape[1]):
            if abs(corr_matrix[i, j]) > threshold:
                retain[j] = False

    x_filtered = x[:, retain]
    feature_names_filtered = np.array(feature_names)[retain]
    return x_filtered, feature_names_filtered, retain

def normalisetransform(x):
    scalerobject = StandardScaler()
    x_normalised = scalerobject.fit_transform(x)
    return x_normalised, scalerobject

# for the above, vf, retain, and scalerobject are also returned so that they can be used on the test or validation data

def mann_whitney_test(x, y):
    group1 = x[y == 0] # patients where non pcr
    group2 = x[y == 1] # patients where pcr

    # SelectKBest requires there to be a function that returns scores and pvalues
    scores = np.empty(x.shape[1]) # number of features
    pvalues = np.empty(x.shape[1])

    for feature_num in range(x.shape[1]):
        ustat, pvalue = mannwhitneyu(group1[:, feature_num], group2[:, feature_num])
        scores[feature_num] = abs(ustat - (len(group1)*len(group2))/2)
        # quick description of how the score works: a feature is compared against all patients in the OTHER group. e.g. a feature in pcr patient is compared to the same feature in all the non pcr patients
        # hence len(group1)*len(group2). 1/2 because if there is no correlation, then the score would be near 0. this is because the two groups would overlap randomly, and only about half of the comparisons would be saying "yes" to the feature 
        # being higher than the other (which is the purpose of the mannwhitneyu test)
        pvalues[feature_num] = pvalue

    return scores, pvalues

def selectkbestfilter(x, y, feature_names, k: int=5):
    selector = SelectKBest(mann_whitney_test, k=k) # Zhang et al
    x_filtered = selector.fit_transform(x, y)
    feature_names_filtered = np.array(feature_names)[selector.get_support()]
    return x_filtered, feature_names_filtered, selector

def filter_test_data(x, vf_object, retain_matrix, selector_object, scaler_object): #the following must be done in the same order as it was done in the training data
    x_filtered = vf_object.transform(x)
    x_filtered = x_filtered[:, retain_matrix]
    x_filtered = selector_object.transform(x_filtered)
    x_filtered = scaler_object.transform(x_filtered)
    return x_filtered

def filter_train_data(x, y, feature_names, k):
    x_filtered, feature_names_filtered, vf = variancefilter(x, feature_names)
    x_filtered, feature_names_filtered, retain = pearsonfilter(x_filtered, feature_names_filtered)
    x_filtered, feature_names_filtered, selector = selectkbestfilter(x_filtered, y, feature_names_filtered, k=k)
    x_filtered, scalerobject = normalisetransform(x_filtered)
    return x_filtered, feature_names_filtered, vf, retain, selector, scalerobject # feed vf, retain, scalerobject directly into filter_test_data

def phase_number(path):
    return int(path.stem.rsplit("_", 1)[-1])

def get_patient_id(path):
    return str(path.stem.rsplit("_", 1)[0])

def dataset_loader(dataset_name: str, num_phases: int, datapath: Path, clinpath: Path): # clinroot is the path to the sheet that contains all the clinical data, including pcr. dataset is the dataset name e.g. NACT for the NACT pilot
    # this function needs to pull the data, and save it into something that can plug into the featurevectors.py classes.
    # this means that patient: {'images': all phase images {"phase_0": image_0, etc etc}, 'mask': mask image, 'pcr': 0 or 1, }
    # each image and mask must be a simple itk Image object
    dataset = defaultdict()
    clinical_information = pd.read_excel(clinpath, index_col=0)
    for patient in datapath.glob(f"{dataset_name}_*"):
        patient_info = defaultdict()
        patient_id = get_patient_id(patient) # example output would be "NACT_01"

        phase_paths = sorted(patient.glob(f"{patient_id}_*.nii"), key=phase_number) # ensures that they are sorted based on phase
        if len(phase_paths) != num_phases:
            continue

        patient_info['images'] = defaultdict()
        for phase_path in phase_paths:
            patient_info['images'][str(phase_number(phase_path))] = sitk.ReadImage(str(phase_path))

        patient_info['mask'] = sitk.ReadImage(str(patient / 'mask.nii'))

        patient_info['pcr'] = clinical_information.loc[patient_id, "pcr"]

        patient_info['time_res'] = ast.literal_eval(clinical_information.loc[patient_id, "acquisition_times"])

        dataset[patient_id] = patient_info

    return dataset

def n4_correction(image): # image being a simple itk readimage object
    image = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrected_image = corrector.Execute(image, mask)
    return corrected_image

def resample(image, is_mask: bool=False, new_spacing: tuple=(1.0, 1.0, 1.0)): # image (or mask) being simple itk readimage object
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    resampler = sitk.ResampleImageFilter()
    resampled_size = [int(round(size * spacing / new_spacing_value)) for size, spacing, new_spacing_value in zip(original_size, original_spacing, new_spacing)]

    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(resampled_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # the reason for using this is in the sitk resampleimagefilter documentation
        resampler.SetOutputPixelType(sitk.sitkUInt8) # binary ints
    else:
        resampler.SetInterpolator(sitk.sitkLinear) # this should be the default for the sitk interpolator, but just in case
        resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(image)

def z_score_image_normalise(images: Dict): # images are stored as a dictionary under 'image' in the patient 
    all_pixels = np.concatenate([sitk.GetArrayViewFromImage(image).ravel() for image in images.values()])
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)

    if std == 0:
        raise ValueError("std cannot be 0")

    normalised = {}
    for phase_name, image in images.items():
        image_values = sitk.Cast(image, sitk.sitkFloat32)
        normalised[phase_name] = (image_values - float(mean))/float(std)

    return normalised

def data_preprocess(dataset):
    # dataset is the dictionary with all the patients in it as keys
    for patient, patient_info in dataset.items():
        images = patient_info['images']
        mask = patient_info['mask']
        corrected_images = {phase_name: n4_correction(image) for phase_name, image in images.items()}
        resampled_images = {phase_name: resample(image) for phase_name, image in corrected_images.items()}
        r_mask = resample(mask, is_mask=True)

        patient_info['images'] = z_score_image_normalise(resampled_images)
        patient_info['mask'] = r_mask

    return dataset
