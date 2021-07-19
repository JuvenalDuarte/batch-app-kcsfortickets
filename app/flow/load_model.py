from ..functions.update_embeddings import update_embeddings
from .commons import Task
import luigi
from . import ingestion
from pycarol.pipeline import inherit_list
from sentence_transformers import SentenceTransformer
from pycarol.pipeline.targets import PickleTarget, PytorchTarget
import logging
import torch

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

class LoadModel(Task):
    #model_name = luigi.Parameter()
    #model_version = luigi.Parameter()
    
    #target_type = PickleTarget
    target_type = PytorchTarget

    def easy_run(self, inputs):

        try:
            gpu = torch.cuda.is_available()
            logger.info(f'GPU enabled? {gpu}.')
            if gpu:
                logger.info(f'GPU model: {torch.cuda.get_device_name(0)}.')
            else:
                logger.info(f'Running on CPU mode.')
        except Exception as e:
            logger.error(f'Cannot verify if GPU is available: {e}.')

        logger.info(f'Loading embedding model.')
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

        return model
