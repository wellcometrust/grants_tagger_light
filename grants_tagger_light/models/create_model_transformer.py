import ast
import logging

from grants_tagger_light.models.bert_mesh import WellcomeBertMesh
from grants_tagger_light.utils import load_pickle

logger = logging.getLogger(__name__)


def create_model(parameters=None):
    model = WellcomeBertMesh()

    if parameters:
        params = ast.literal_eval(parameters)
        model.set_params(**params)
    else:
        parameters = {}
    return model


def load_model(model_path, parameters=None):
    if str(model_path).endswith(".pkl"):
        model = load_pickle(model_path)
    else:
        model = create_model(parameters=parameters)
        model.load(model_path)

    return model
