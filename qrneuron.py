import json
import random
import string
import argparse
from pathlib import Path
from keyneuron import KeyNeuron

def parse_args():
    parser = argparse.ArgumentParser(description="KeyNeuron extraction script")

    parser.add_argument("--json-path", type=str, required=True,
                        help="Path to the input JSON file containing data samples.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path or name of the model (e.g., Hugging Face model ID).")
    parser.add_argument("--result-dir", type=str, default="results/",
                        help="Directory to save the extracted key neurons.")
    parser.add_argument("--common-threshold", type=float, default=0.3,
                        help="Threshold for selecting common neurons.")
    parser.add_argument("--top-v", type=int, default=20,
                        help="Number of top neurons to consider per sample.")
    parser.add_argument("--attr-threshold", type=float, default=0.2,
                        help="Attribution score threshold for selecting neurons.")
    parser.add_argument("--num-labels", type=int, default=4,
                        help="Number of possible answer options (e.g., 4 for multiple-choice).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for model inference.")
    parser.add_argument("--steps", type=int, default=16,
                        help="Number of steps for neuron attribution.")

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()

    random.seed(42)

    DATA_PATH = Path(args.json_path)#"mmlu_random_uuid_sample.json")#"data/domain_sample_multi_choice_qa.json")
    DATA_SAMPLE = json.load(open(DATA_PATH))
    print("[ sample number = {a}]".format(a=len(DATA_SAMPLE)))

    # print(DATA_SAMPLE[0])

    KN = KeyNeuron(
        model_name = args.model_path,#"Qwen/Qwen2.5-0.5B-Instruct",
        data_samples = DATA_SAMPLE,
        result_dir = args.result_dir,#'data/',
        common_threshold=args.common_threshold,#0.3,
        top_v=args.top_v,#20, 
        attr_threshold=args.attr_threshold,#0.2,
        option_letters = list(string.ascii_uppercase)[:args.num_labels)],#["A", "B", "C", "D"],
        batch_size = args.batch_size,#8,
        steps = args.steps,#16,
    )
    # extract key neurons and store them in the result_dir
    KN._extract_key_neuron()

if __name__ == "__main__":
    main()
