from __future__ import annotations
import scipy.stats
import numpy as np
from collections import defaultdict
import pycatch22
import torch
import torch.nn as nn
#import torchvision.models.video as video_models
import pysera
import SimpleITK as sitk

class StaticRadiomics:
    def __init__(self, extractor):
        self.extractor = extractor
        self.features = None
    def extract(self, image, mask):
        self.features = self.extractor.execute(image, mask)
        self.features = {fname: float(value) for fname, value in self.features.items() if not fname.startswith("diagnostics_")}
        return self.features
    def get_feature_vector(self):
        return list(self.get_features().values()), list(self.get_features().keys())
    def get_features(self): # so they don't need to extract features over and over
        if self.features is not None:
            return self.features


class MultiphaseRadiomics: # if using S + M then use only M
    # reminder that this class works using a deep static radiomics as well. if that is passed instead, this is DM rather than M
    def __init__(self, single_pr: StaticRadiomics | DeepStaticRadiomics):
        self.single_pr = single_pr 
        self.features = None
    def extract(self, images, mask): # images being a dict of the phases, in order of phases
        self.features = defaultdict() # pyradiomics outputs a dictionary
        for key, value in images.items():
            phase_features = self.single_pr.extract(value, mask)
            self.features[key] = phase_features
        #shape: features = {0: { }, 1: { }, 2: { }, etc.}
        return self.features

    def get_feature_vector(self):
        output = []
        phase_names = list(self.features.keys())
        feature_names = list(self.features[phase_names[0]].keys())
        multiphase_feature_names = []
        for phase_name in phase_names:
            phase_features = self.features[phase_name]
            output.extend(phase_features[n] for n in feature_names)
            for feature_name in feature_names:
                multiphase_feature_names.append(f"{phase_name}_{feature_name}")
        #shape: output = [f, f, f, f, f, f, f, etc.]
        assert list(phase_features.keys()) == feature_names
        return np.array(output, dtype=np.float32), multiphase_feature_names
            
    def create_time_series_table(self): 
        # this should have shape: Row (different features) Column (features from diff time phases)
        features = self.get_features() # this being a dictionary with shape features = {0: { }, 1: { }, 2: { }, etc.}
        phase_names = list(features.keys())
        feature_names = list(features[phase_names[0]].keys())

        feature_names = [name for name in feature_names if not name.startswith("original_shape_")] # shape will likely not change much across the time series. However, check this statistically. only impacts E and T

        output_table = []
        for phase_name in phase_names:
            phase_features = features[phase_name] #reminder this is a dict
            row = [phase_features[n] for n in feature_names] #row containing values from one time phase
            output_table.append(row)
        #output_table is now of shape Column(features from 1 time phase) Row (diff time phases)
        output_table = np.array(output_table).T #shape Row(different features) Column (features from diff time phases)
        return output_table, feature_names

    def get_features(self):
        if self.features is not None:
            return self.features


class TimeSeriesRadiomics: # Note: the point of calling multi_pr is so that features can be extracted once per data, rather than for M, E, and T separately
    def __init__(self, multi_pr: MultiphaseRadiomics):
        '''Input of catch22 would be 1 radiomic feature across all time phases e.g. [R@TP1, R@TP2, R@TP3, R@TP4, etc.]. Do it across all radiomic features. This should be similar to when doing E: create a table per patient, and iterate through the entire thing (but for E it creates m and b).
        Feature selection and filtering must be done within the training loop (since otherwise there would be 1000s of features in one patient's feature vector)
        Output of catch22 is a dict, similar to that of pyradiomics
        Catch22 has been used for other thighs like fMRI before as well. Get these papers. '''
        self.multi_pr = multi_pr

    def get_feature_vector(self):

        ts_table, feature_names = self.multi_pr.create_time_series_table() #multi_pr must have extracted before this
        feature_vector = []
        for i in ts_table: #iterate through the rows
            feature_vector_component = pycatch22.catch22_all(i) #outputs a dictionary {'names': [], 'values': []}
            feature_vector.extend(feature_vector_component["values"])
        timeseries_feature_names = []
        for feature_name in feature_names:
            timeseries_feature_names.extend([f"{feature_name}_feature1", f"{feature_name}_feature2", f"{feature_name}_feature3", f"{feature_name}_feature4", f"{feature_name}_feature5", f"{feature_name}_feature6", f"{feature_name}_feature7", f"{feature_name}_feature8", f"{feature_name}_feature9", f"{feature_name}_feature10", f"{feature_name}_feature11", f"{feature_name}_feature12", f"{feature_name}_feature13", f"{feature_name}_feature14", f"{feature_name}_feature15", f"{feature_name}_feature16", f"{feature_name}_feature17", f"{feature_name}_feature18", f"{feature_name}_feature19", f"{feature_name}_feature20", f"{feature_name}_feature21", f"{feature_name}_feature22"])
        return feature_vector, timeseries_feature_names

    
class EngineeredTempRadiomics:
    def __init__(self, multi_pr: MultiphaseRadiomics, time_res: list=None):
        self.multi_pr = multi_pr
        self.time_res = time_res
    def get_feature_vector(self):
        if self.time_res is None:
            raise ValueError("time_res not provided")
        ts_table, feature_names = self.multi_pr.create_time_series_table()
        feature_vector = []
        for i in ts_table: # i would be the y value, time_res as x value
            result = scipy.stats.linregress(self.time_res, i)
            feature_vector.extend([result.slope, result.intercept])
        engineered_feature_names = []
        for feature_name in feature_names:
            engineered_feature_names.extend([f"{feature_name}_slope", f"{feature_name}_intercept"])
        feature_vector = np.array(feature_vector, dtype=np.float32)
        return feature_vector, engineered_feature_names
    def apply_time_res(self, tr):
        self.time_res = tr
    


#P excluding PK unless enough temporal resolution
class ConvRadiomics:
    def __init__(self, time_res: list): 

        self.time_res = time_res
        #time_res is list of times [1, 3, 5]s
        
class DeepStaticRadiomics:
    def __init__(self, encoder=pysera):
        self.encoder = encoder
        self.features = None

    def extract(self, image, mask, resultpath:str="./pysera_output", ep: bool=False):
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)
        results = self.encoder.process_batch(image_input=image_array, mask_input=mask_array, output_path=resultpath, extraction_mode='deep_feature', deep_learning_model='resnet50', enable_parallelism=ep)
        # process_batch returns a dictionary. features in 'features_extracted'
        if not results['success']:
            raise RuntimeError(results['error'])

        features_extracted = results['features_extracted']
        features_nums = features_extracted.select_dtypes(include=np.number)

        self.features = {fname: float(feature) for fname, feature in features_nums.iloc[0].to_dict().items()}

        return self.features

    def get_feature_vector(self):
        return list(self.get_features().values()), list(self.get_features().keys())

    def get_features(self):
        if self.features is not None:
            return self.features
    
class DeepTempRadiomics:
    def __init__(self):
        pass

