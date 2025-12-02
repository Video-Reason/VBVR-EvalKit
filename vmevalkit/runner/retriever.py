
from typing import List, Dict, Any, Tuple, Optional

from vmevalkit.runner.dataset import (
    create_vmeval_dataset_direct,
    download_hf_domain_to_folders
)
from vmevalkit.runner.TASK_CATALOG import TASK_REGISTRY


class Retriever:
    """Retriever class for managing domain selection and dataset creation."""
    
    def __init__(
        self,
        dataset_config: List[Dict[str, Any]]
    ):

        self.output_path = dataset_config.get('output_path', 'data/questions')
        self.task_name = dataset_config.get('task', None)
        self.pairs_per_domain = dataset_config.get('pairs_per_domain', 5)
        self.random_seed = dataset_config.get('random_seed', 42)
        self.config = TASK_REGISTRY.get(self.task_name, {})
    
    def download_hf_domains(self) -> None:
        """
        Download HuggingFace domains to folder structure.
        """

        download_hf_domain_to_folders(self.task_name, self.output_path)
    
    def create_regular_dataset(self) -> Tuple[Dict[str, Any], str]:
        """
        Create dataset for regular (non-HuggingFace) domains.
        
        Returns:
            Tuple of (dataset dictionary, path to questions directory)
        """
        dataset, questions_dir = create_vmeval_dataset_direct(
            pairs_per_domain=self.pairs_per_domain,
            random_seed=self.random_seed,
            selected_tasks=[self.task_name]
        )
        return dataset, questions_dir
    
    def retrieve_tasks(self) -> List[Dict[str, Any]]:
        if self.config.get('hf', False):
            self.download_hf_domains()
        else:
            self.create_regular_dataset()

