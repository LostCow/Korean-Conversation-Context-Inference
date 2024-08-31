import random

import numpy as np
import torch
from transformers import set_seed
from transformers.trainer_utils import EvalPrediction

category_map = {
    "원인": "대화에 일어난 사건을 유발하는 사건",
    "후행사건": "대화 이후에 일어날 수 있는 사건",
    "전제": "대화에 일어난 사건을 가능하게 하는 상태 혹은 사건",
    "동기": "대화를 일으키는 '화자'의 감정이나 기본 욕구",
    "반응": "대화에 일어난 사건에 대해 '청자'가 보일 수 있는 감정 반응",
}
output_map = {"inference_1": "A", "inference_2": "B", "inference_3": "C"}
id_to_category = {0: "원인", 1: "후행사건", 2: "전제", 3: "동기", 4: "반응"}
category_to_id = {v: k for k, v in id_to_category.items()}
id_to_inference = {0: "inference_1", 1: "inference_2", 2: "inference_3"}


def seed_everything(random_seed: int) -> None:
    set_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
