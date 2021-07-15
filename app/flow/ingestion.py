from ..functions.ingestion import data_ingestion
from ..flow.commons import Task
import luigi
import pandas as pd
from . import merge
from pycarol.pipeline import inherit_list
import logging

logger = logging.getLogger(__name__)
luigi.auto_namespace(scope=__name__)

@inherit_list(
    merge.MergeDocuments
)
class IngestDocuments(Task):
    staging_name = luigi.Parameter() 
    daystopresent = luigi.IntParameter()

    def easy_run(self, inputs):
        recent_art = data_ingestion(stag=self.staging_name, daystopresent=self.daystopresent)
        return recent_art