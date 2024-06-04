#!/usr/local/bin/python3

import os
import log
import torch
import random
import argparse
import numpy as np
from trainer import *
from load_datasets import *


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def main():
    parser = argparse.ArgumentParser(description="Command line interface for finding multi-modal neurons.")
    parser.add_argument("--model_type", default="LLaVA", type=str, help="Type of model")
    parser.add_argument("--model_base", default=None, type=str, help="Base of model")
    parser.add_argument("--model_path", type=str, default="../../model/llava-llama-2-13b-chat-lightning-preview", help="Path to model")
    parser.add_argument("--save_to", type=str, default="../../results")
    parser.add_argument("--task_type", default="sbu", type=str, help="Type of task")
    parser.add_argument("--data_path", default='../../datasets/sbu',
                        type=str, help="Path to the data")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed used")
    parser.add_argument("--bz", default=1, type=int, help="Batch size")
    parser.add_argument("--max_num", default=1000, type=int, help="Max number of executing images")
    parser.add_argument("--start", default=0, type=int, help="No. of the start image")
    parser.add_argument("--end", default=1000, type=int, help="No. of the end image")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--add_prompt", action="store_true", help="Add prompt or not")
    parser.add_argument("--cal_activations", action="store_true")
    parser.add_argument("--shuffle", action="store_true")

    args = parser.parse_args()

    # set seed
    seed = args.random_seed
    print("Task type:", args.task_type)
    print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args.task_type = args.task_type.lower()
    args.model_type = args.model_type.lower()

    if args.task_type == "sbu":
        train_df, valid_df, test_df = load_sbu(args.data_path)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train = SBU(train_df, args.data_path, transform=transform)
        valid = SBU(valid_df, args.data_path, transform=transform)
        test = SBU(test_df, args.data_path, transform=transform)

        args.query = "Describe the image in few words."

    else:
        raise ValueError(f"Unknown task type: {args.task_type}")

    args.save_to += f"/{args.task_type}/find_mm_neurons"

    print(args.query)

    create_dir(args.save_to)

    t = trainer(args)

    t.find_mm_neurons(train)


if __name__ == "__main__":
    main()
