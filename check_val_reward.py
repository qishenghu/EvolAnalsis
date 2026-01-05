import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="analysis/experiments/alfworld")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args.output_file)
    val_data = []
    with open(args.output_file, "r") as f:
        val_data = [json.loads(line) for line in f]
    avg_reward = sum([x['reward'] for x in val_data]) / len(val_data)
    print(f"Average reward: {avg_reward}, Length: {len(val_data)}")
