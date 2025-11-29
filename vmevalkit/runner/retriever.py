from pathlib import Path
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
        output_dir: str = "data/questions",
        tasks: Optional[List[str]] = None,
        random_seed: Optional[int] = 42,
        pairs_per_domain: int = 50
    ):
        self.output_path = Path(output_dir)
        self.pairs_per_domain = pairs_per_domain
        self.random_seed = random_seed
        self.tasks = tasks
        self.hf_domains = [
            d for d in self.tasks 
            if TASK_REGISTRY.get(d, {}).get('hf', False) is True 
            and not TASK_REGISTRY.get(d, {}).get('hf_meta', False)
        ]
        self.regular_domains = [
            d for d in self.tasks
            if TASK_REGISTRY.get(d, {}).get('hf', False) is not True
        ]
    
    def download_hf_domains(self) -> None:
        """
        Download HuggingFace domains to folder structure.
        """
        if self.hf_domains:
            for domain in self.hf_domains:
                download_hf_domain_to_folders(domain, self.output_path)
    
    def create_regular_dataset(self) -> Tuple[Dict[str, Any], str]:
        """
        Create dataset for regular (non-HuggingFace) domains.
        
        Returns:
            Tuple of (dataset dictionary, path to questions directory)
        """
        if self.regular_domains:
            dataset, questions_dir = create_vmeval_dataset_direct(
                pairs_per_domain=self.pairs_per_domain,
                random_seed=self.random_seed,
                selected_tasks=self.regular_domains
            )
        
        return dataset, questions_dir
