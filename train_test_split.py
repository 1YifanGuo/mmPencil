import os
import json
import random
from collections import defaultdict


def read_jsonl(file_path):
    """Read a JSONL file and return a list of records."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_jsonl(records, file_path):
    """Write a list of records to a JSONL file."""
    with open(file_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def split_data_by_label(records, train_ratio=0.8):
    """Split records by label into train and test sets based on the given ratio."""
    label_to_records = defaultdict(list)

    # Group records by label
    for record in records:
        label = record['messages'][1]['content']
        label_to_records[label].append(record)

    train_set = []
    test_set = []

    # Split each group into train and test sets
    for label, records in label_to_records.items():
        random.shuffle(records)
        split_index = int(len(records) * train_ratio)
        train_set.extend(records[:split_index])
        test_set.extend(records[split_index:])

    return train_set, test_set


def main():
    # Define the paths to the JSONL files
    jsonl_files = [
        "mmPencil_dataset/text/User-01/200-Word.jsonl",
        "mmPencil_dataset/text/User-02/200-Word.jsonl",
        "mmPencil_dataset/text/User-03/200-Word.jsonl",
        "mmPencil_dataset/text/User-04/200-Word.jsonl"
    ]

    # Read all records from the JSONL files
    all_records = []
    for file_path in jsonl_files:
        all_records.extend(read_jsonl(file_path))

    # Split the records into train and test sets
    train_set, test_set = split_data_by_label(all_records, train_ratio=0.8)

    # Write the train and test sets to new JSONL files
    train_output_path = "mmPencil_dataset/text/train.jsonl"
    test_output_path = "mmPencil_dataset/text/test.jsonl"

    write_jsonl(train_set, train_output_path)
    write_jsonl(test_set, test_output_path)

    print(f"Train set written to {train_output_path}")
    print(f"Test set written to {test_output_path}")


if __name__ == "__main__":
    main()