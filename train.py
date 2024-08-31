import logging
import os
import warnings

import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
)
from transformers.trainer_utils import is_main_process

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from prompt import INPUT_PROMPT, TOTAL_PROMPT
from utils import category_map, output_map, seed_everything

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


def main(model_args: ModelArguments, data_args: DatasetsArguments, training_args: MyTrainingArguments):
    seed_everything(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_size="right")
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("json", data_files=data_args.train_data_path, split="train")

    if training_args.do_aug:
        train_aug_dataset = load_dataset("json", data_files=data_args.aug_data_path, split="train")
        train_dataset = concatenate_datasets([train_dataset, train_aug_dataset])

    valid_dataset = load_dataset("json", data_files=data_args.valid_data_path, split="train")

    def preprocess_data(raw):
        category = raw["input"]["category"]
        inference_1 = raw["input"]["inference_1"]
        inference_2 = raw["input"]["inference_2"]
        inference_3 = raw["input"]["inference_3"]
        dialog_info_list = raw["input"]["conversation"]
        dialog_list = []
        for dialog_info in dialog_info_list:
            ref = dialog_info["utterance_id"].split(".")[-1]
            speaker = dialog_info["speaker"]
            utterance = dialog_info["utterance"].replace("\n", " ")
            dialog_list.append(f"화자{speaker}: {utterance}")
        ref_ids = []
        for i, ref_id in enumerate(raw["input"]["reference_id"]):
            tmp = int(ref_id.split(".")[-1])
            ref_ids.append(f"{dialog_list[tmp - 1]}")

        raw["reference_id"] = "\n".join(ref_ids)
        raw["dialog"] = "\n".join(dialog_list)
        raw["inference_1"] = inference_1
        raw["inference_2"] = inference_2
        raw["inference_3"] = inference_3
        raw["category"] = category

        return raw

    train_dataset = train_dataset.map(preprocess_data, keep_in_memory=True)
    valid_dataset = valid_dataset.map(preprocess_data, keep_in_memory=True)

    def preprocess_prompt(raw):
        dialog = raw["dialog"]
        category = raw["category"]
        inference_1 = raw["inference_1"]
        inference_2 = raw["inference_2"]
        inference_3 = raw["inference_3"]
        ref = raw["reference_id"]
        definition = category_map[raw["category"]]
        full_text = TOTAL_PROMPT.format_map(
            {
                "category": category,
                "dialog": dialog,
                "def": definition,
                "ref": ref,
                "inference_1": inference_1,
                "inference_2": inference_2,
                "inference_3": inference_3,
                "answer": output_map[raw["output"]],
            }
        )
        tokenized_full_prompt = tokenizer(full_text)
        user_prompt = INPUT_PROMPT.format_map(
            {
                "category": category,
                "dialog": dialog,
                "def": definition,
                "ref": ref,
                "inference_1": inference_1,
                "inference_2": inference_2,
                "inference_3": inference_3,
            }
        )

        tokenized_user_prompt = tokenizer(user_prompt)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    train_dataset = train_dataset.map(preprocess_prompt, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(preprocess_prompt, remove_columns=valid_dataset.column_names)
    logger.info(tokenizer.decode(valid_dataset[3]["input_ids"][:]))
    logger.info(valid_dataset[3]["input_ids"][-20:])
    logger.info(valid_dataset[3]["labels"][-20:])
    logger.info(
        f"train size: {len(train_dataset)}\nvalid size: {len(valid_dataset)}\nratio: {len(valid_dataset)/(len(train_dataset) + len(valid_dataset))}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
        attn_implementation="eager" if "gemma" in model_args.model_name_or_path.lower() else "sdpa",
        trust_remote_code=True if "EXA" in model_args.model_name_or_path.lower() else "sdpa",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.target_modules,
        inference_mode=False,
        r=model_args.r,
        lora_alpha=model_args.lora_alpha,
        use_rslora=model_args.use_rslora,
        lora_dropout=model_args.lora_dropout,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["A"], tokenizer.vocab["B"], tokenizer.vocab["C"]]
        logits = logits[:, -2, logit_idx]
        return logits

    acc_metric = evaluate.load("accuracy")

    int_output_map = {"A": 0, "B": 1, "C": 2}

    def compute_metrics(evaluation_result):
        logits, labels = evaluation_result

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: int_output_map[x], labels))

        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)
        return acc

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if training_args.local_rank == 0:
        model = model.merge_and_unload()
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, data_args=data_args, training_args=training_args)
