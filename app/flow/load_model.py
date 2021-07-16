from ..functions.update_embeddings import update_embeddings
from .commons import Task
import luigi
from . import ingestion
from pycarol.pipeline import inherit_list
from sentence_transformers import SentenceTransformer
from pycarol.pipeline.targets import PickleTarget, PytorchTarget
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

class LoadModel(Task):
    model_name = luigi.Parameter()
    model_version = luigi.Parameter()
    
    #target_type = PickleTarget
    target_type = PytorchTarget

    def easy_run(self, inputs):
        logger.info(f'Loading embedding model.')
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

        return model
