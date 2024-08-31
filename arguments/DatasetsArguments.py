from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DatasetsArguments:
    train_data_path: str = field(
        default=None,
    )
    valid_data_path: str = field(
        default=None,
    )
    aug_data_path: str = field(
        default=None,
    )
