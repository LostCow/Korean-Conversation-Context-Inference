from dataclasses import dataclass

from transformers import TrainingArguments, Seq2SeqTrainingArguments
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MyTrainingArguments(TrainingArguments):
    do_aug: Optional[bool] = field(default=True)
    pass
