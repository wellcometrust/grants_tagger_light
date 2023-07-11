from transformers import AutoModel
from transformers import AutoTokenizer
import argparse


def save_to_local_from_hub(key: str, out_path: str):
    model = AutoModel.from_pretrained(key, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(key)

    # Save model and tokenizer to file
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        type=str,
        help="The key of the model to save locally",
        default="Wellcome/WellcomeBertMesh",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="The path to save the model to",
        default="models/WellcomeBertMesh-fromhub",
    )
    args = parser.parse_args()

    print(args)

    save_to_local_from_hub(args.key, args.out_path)
