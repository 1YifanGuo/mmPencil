import json
import argparse
from collections import defaultdict

def calculate_accuracy_and_per_label(jsonl_path):
    total = 0
    correct = 0
    label_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            response = str(example.get('response', '')).strip().lower()
            label = str(example.get('labels', '')).strip().lower()

            total += 1
            is_correct = response == label
            correct += int(is_correct)
            label_stats[label]['total'] += 1
            label_stats[label]['correct'] += int(is_correct)

    overall_accuracy = correct / total if total > 0 else 0.0
    print(f"\n===== Overall Accuracy =====")
    print(f"Total samples        : {total}")
    print(f"Correct predictions  : {correct}")
    print(f"Overall Accuracy     : {overall_accuracy:.4f}\n")

    label_accuracy = []
    for label, stats in label_stats.items():
        l_total = stats['total']
        l_correct = stats['correct']
        acc = l_correct / l_total if l_total > 0 else 0.0
        label_accuracy.append((label, l_total, l_correct, acc))

    label_accuracy.sort(key=lambda x: x[3])

    print(f"===== Per-label Accuracy (sorted by accuracy) =====")
    print(f"{'Label':<20} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 60)
    for label, total, correct, acc in label_accuracy:
        print(f"{label:<20} {total:<10} {correct:<10} {acc:<.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute overall and per-label accuracy from a JSONL file comparing 'response' and 'labels'.")
    parser.add_argument("jsonl_path", type=str, help="Path to the JSONL file.")
    args = parser.parse_args()

    calculate_accuracy_and_per_label(args.jsonl_path)