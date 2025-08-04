import json
from pathlib import Path


def convert_jsonl_to_json(input_path: str, output_path: str):
    lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(lines, f, indent=2)


if __name__ == "__main__":
    input_file = Path("../../data/News_Category_Dataset_v3.json")
    output_file = Path("../data/converted_dataset.json")
    convert_jsonl_to_json(input_file, output_file)
    print(f"✅ Converted {input_file} to {output_file}")
