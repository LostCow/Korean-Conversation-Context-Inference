import argparse
import copy
import json


def main(args):
    aug = []

    with open(args.data_path, "r") as f:
        data = json.load(f)

    for d in data:
        origin_label = d["output"]
        inference_1 = d["input"]["inference_1"]
        inference_2 = d["input"]["inference_2"]
        inference_3 = d["input"]["inference_3"]

        inference_idx = set([0, 1, 2])
        inference_total = [inference_1, inference_2, inference_3]
        correct_idx = int(origin_label.split("_")[-1]) - 1

        correct_sent = inference_total[correct_idx]

        wrong_idxs = list(inference_idx.difference([correct_idx]))

        for wrong_idx in wrong_idxs:
            tmp = copy.deepcopy(d)
            remain_idx = list(inference_idx.difference([correct_idx, wrong_idx]))[0]

            tmp["input"][f"inference_{correct_idx+1}"] = inference_total[wrong_idx]
            tmp["input"][f"inference_{wrong_idx+1}"] = correct_sent
            tmp["input"][f"inference_{remain_idx+1}"] = inference_total[remain_idx]
            tmp["output"] = f"inference_{wrong_idx+1}"
            aug.append(tmp)

        with open(args.save_path, "w") as f:
            f.write(json.dumps(aug, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    main(args)
