import numpy as np
from .dualdict import *
from ..core.ml.classification import PartitionedClassifier, XClassifier
from ..core.ml.regression import PartitionedRegressor, XRegressor
from .encoders import profile_parse
import json


def parse_classifier_response(response, model=None):
    if response['model_type'] != 'binary_classification':
        raise ValueError(f'Model is not a binary classification model')

    if model is None:
        partitioned_model = PartitionedClassifier(response['partition_on'])
    else:
        partitioned_model = model

    for p in response['partitions']:
        model = XClassifier()

        model._profile = np.array([
            np.array(i) for i in json.loads(p['profile'])], dtype=object)

        model._profile = profile_parse(model._profile)

        model._calibration_map = {
            int(i): v for i, v in p['calibration_map'].items()}

        model._support_map = {
            int(i): v for i, v in p['support_map'].items()}

        model.base_value = p['base_value']
        model.target_map = TargetMap({int(i): v for i, v in p['target_map'].items()}, True)
        model.feature_map = {k: FeatureMap(v) for k, v in p['feature_map'].items()}

        model.columns = p['columns']
        model.id_columns = p['id_columns']

        model.categorical_columns = p['feature_map'].keys()
        model.numeric_columns = [c for c in model.columns if c not in model.categorical_columns]

        if 'constructs' in p:
            model.constructs_from_json(p['constructs'])

        model.category_meta = {
            i: {ii: {int(float(k)): v for k, v in vv.items()} for ii, vv \
                in v.items()} for i, v in p['category_meta'].items()}

        partitioned_model.add_partition(model, p['partition'])

    return partitioned_model

def parse_regressor_response(response, model=None):
    if model is None:
        partitioned_model = PartitionedRegressor(response['partition_on'])
    else:
        partitioned_model = model

    for p in response['partitions']:
        model = XRegressor()
        model._profile = np.array([np.array(i) for i in json.loads(p['profile'])])
        model._profile = profile_parse(model._profile)
        model.base_value = p['base_value']
        model.feature_map = {k: FeatureMap(v) for k, v in p['feature_map'].items()}
        # model.parameters = ConstructorParams(p['parameters'])

        model.columns = p['columns']
        model.id_columns = p['id_columns']

        model.categorical_columns = p['feature_map'].keys()
        model.numeric_columns = [c for c in model.columns if c not in model.categorical_columns]

        if 'constructs' in p:
            model.constructs_from_json(p['constructs'])

        model.category_meta = {
            i: {ii: {int(float(k)): v for k, v in vv.items()} for ii, vv \
                in v.items()} for i, v in p['category_meta'].items()}

        partitioned_model.add_partition(model, p['partition'])

    return partitioned_model