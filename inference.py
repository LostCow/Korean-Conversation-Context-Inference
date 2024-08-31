import argparse
import json
import os

import numpy as np
import torch
from accelerate import infer_auto_device_map
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import INPUT_PROMPT
from utils import category_map, id_to_inference


def preprocess(data):
    for raw in data:
        category = raw["input"]["category"]
        inference_1 = raw["input"]["inference_1"]
        inference_2 = raw["input"]["inference_2"]
        inference_3 = raw["input"]["inference_3"]
        dialog_info_list = raw["input"]["conversation"]
        dialog_list = []
        for dialog_info in dialog_info_list:
            ref = dialog_info["utterance_id"].split(".")[-1]
            speaker = dialog_info["speaker"]
            utterance = f'{dialog_info["utterance"]}'.replace("\n", " ")
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

    return data


def main(args):
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_map = infer_auto_device_map(model, max_memory={0: "22GiB", "cpu": "96GiB"})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="eager" if "gemma" in model_path else "sdpa",
    )

    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    preprocess_data = preprocess(data)

    answers = []
    with torch.inference_mode():
        for raw in tqdm(preprocess_data):
            input_text = INPUT_PROMPT.format_map(
                {
                    "category": raw["category"],
                    "dialog": raw["dialog"],
                    "ref": raw["reference_id"],
                    "def": category_map[raw["category"]],
                    "inference_1": raw["inference_1"],
                    "inference_2": raw["inference_2"],
                    "inference_3": raw["inference_3"],
                }
            )
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model(inputs["input_ids"].to("cuda"))
            logits = outputs.logits[:, -1].flatten().cpu()
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [logits[tokenizer.vocab["A"]], logits[tokenizer.vocab["B"]], logits[tokenizer.vocab["C"]]]
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            answers.append(probs)
            torch.cuda.empty_cache()

    answers = np.vstack(answers)
    preds = np.argmax(answers, axis=1)

    with open(args.data_path, "r") as f:
        data = json.load(f)  # 제출용으로 다시 로드

    for d, a in zip(data, preds):
        d["output"] = id_to_inference[a]

    if not os.path.isdir("output"):
        os.mkdir("outputs")
    with open(os.path.join("outputs", args.output_path), "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/test.json", type=str)
    parser.add_argument("--model_path", default="models/x2bee/POLAR-14B-v0.5", type=str)
    parser.add_argument("--output_path", default="submission.json", type=str)
    args = parser.parse_args()

    main(args)
