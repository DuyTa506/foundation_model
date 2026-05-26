"""
Usage:
    python update_yaml_sampling.py input.yaml --output_yaml updated.yaml \
         --text_percent 14 --image_percent 50 --multiimage_percent 5 --video_percent 36 \
         --short_video_factor 0.5 --mammoth_fraction 0.3
"""

import yaml
import argparse
import re
import sys
from collections import defaultdict

# -------------------------------------------------------------------
# Helper functions for sampling logic.
# -------------------------------------------------------------------

def always_full(dataset_entry):
    """
    Return True if the dataset entry should always be sampled at 100%.
    We check for keywords like "gpt", "visualwebinstruct", or "vision_flan".
    """
    always_full_keywords = ["gpt", "visualwebinstruct", "vision_flan"]
    for field in [dataset_entry.get("name", ""), dataset_entry.get("path", ""), dataset_entry.get("json_path", "")]:
        if any(kw in field.lower() for kw in always_full_keywords):
            return True
    return False

def base_sampling_fraction(dataset_entry, text_percent, image_percent, multiimage_percent, video_percent, short_video_factor):
    """
    Compute the overall target fraction (a float in [0,1]) for the dataset entry
    based on its modality and video duration if applicable.
    
    For video datasets, we look for a pattern like _X_Y_s in the name or json_path.
    If found and if the upper bound (Y) is <= 60, then we treat it as a short video.
    """
    modality = dataset_entry.get("modality", "").lower()
    name = dataset_entry.get("name", "").lower()
    json_path = dataset_entry.get("json_path", "").lower()

    if modality == "text":
        return text_percent / 100.0
    elif modality == "video":
        # Look for a pattern like _X_Y_s in the name or json_path.
        pattern = r'_(\d+)_([\d]+)_s'
        match = re.search(pattern, name)
        if not match:
            match = re.search(pattern, json_path)
        if match:
            upper_bound = int(match.group(2))
            if upper_bound <= 60:
                return (video_percent / 100.0) * short_video_factor
            else:
                return video_percent / 100.0
        else:
            # If no pattern is found, default to full video_percent.
            return video_percent / 100.0
    elif modality == "image":
        return image_percent / 100.0
    elif modality == "multiimage" or "multiimage" in name:
        return multiimage_percent / 100.0
    else:
        # If modality is unknown, default to 100%.
        return 1.0

def format_sampling(fraction):
    """
    Convert a fraction (0 to 1) into a sampling_strategy string.
    For example, 0.33 becomes "random:33.0%".
    """
    percent_str = round(fraction * 100, 2)
    return f"random:{percent_str}%"

# -------------------------------------------------------------------
# Functions for aligning datasets using the 'name' field.
# -------------------------------------------------------------------

def compute_aligned_name(dataset_entry):
    """
    Compute an "aligned name" for the dataset entry by looking at its name.
    We expect names to be of the form "mammoth:sharegpt" or "onevision:sharegpt".
    We split on ":" and return the second part as the aligned name.
    If no colon is found, we default to using the full name (lowercased and stripped).
    """
    name = dataset_entry.get("name", "").strip()
    if ':' in name:
        # Assume the format is "source:aligned_name"
        parts = name.split(":", 1)
        return parts[1].strip().lower()
    else:
        return name.lower()

def is_onevision(dataset_entry):
    """
    Return True if the dataset entry is from OneVision. We assume the name starts with "onevision:".
    """
    name = dataset_entry.get("name", "").lower().strip()
    return name.startswith("onevision:")

def is_mammoth(dataset_entry):
    """
    Return True if the dataset entry is from MammothVL. We assume the name starts with "mammoth:".
    """
    name = dataset_entry.get("name", "").lower().strip()
    return name.startswith("mammoth:")

# -------------------------------------------------------------------
# YAML load/write functions.
# -------------------------------------------------------------------

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_yaml(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

# -------------------------------------------------------------------
# Main update logic.
# -------------------------------------------------------------------

def update_sampling_strategies(data, text_percent, image_percent, multiimage_percent, video_percent, short_video_factor, mammoth_fraction):
    """
    Update each dataset entry's sampling_strategy according to these rules:
      - If always_full applies, use "random:100%".
      - Otherwise, compute an overall target fraction using modality rules.
      - For datasets that share the same aligned name (e.g. "sharegpt"), split the target fraction:
            OneVision gets overall_fraction * (1 - mammoth_fraction)
            MammothVL gets overall_fraction * mammoth_fraction
      - For non-shared datasets, simply use the overall fraction.
    """
    if "datasets" not in data or not isinstance(data["datasets"], list):
        print("Error: YAML must contain a top-level 'datasets' list.")
        sys.exit(1)
        
    datasets = data["datasets"]
    
    # Group dataset entries by aligned name.
    groups = defaultdict(list)
    for ds in datasets:
        aligned = compute_aligned_name(ds)
        groups[aligned].append(ds)
    
    # Update each entry.
    for aligned_name, group_entries in groups.items():
        shared = len(group_entries) > 1
        for ds in group_entries:
            if always_full(ds):
                ds["sampling_strategy"] = "random:100%"
                continue
            
            overall_fraction = base_sampling_fraction(ds, text_percent, image_percent, multiimage_percent, video_percent, short_video_factor)
            
            if shared:
                # If both OneVision and MammothVL versions exist for this aligned name,
                # split the overall fraction.
                if is_onevision(ds):
                    final_fraction = overall_fraction * (1 - mammoth_fraction)
                elif is_mammoth(ds):
                    final_fraction = overall_fraction * mammoth_fraction
                else:
                    # If the source is unclear, fall back to overall.
                    final_fraction = overall_fraction
            else:
                final_fraction = overall_fraction
            
            ds["sampling_strategy"] = format_sampling(final_fraction)
    
    return data

def main():
    parser = argparse.ArgumentParser(description=(
        "Update dataset sampling strategies in a YAML file. For datasets that are shared (i.e. have the same aligned name), "
        "split the overall target between the OneVision and MammothVL versions based on --mammoth_fraction."
    ))
    parser.add_argument("input_yaml", help="Path to input YAML file.")
    parser.add_argument("--output_yaml", default="updated_datasets.yaml", help="Path to output YAML file.")
    parser.add_argument("--text_percent", type=float, default=14.0, help="Target percentage for text datasets.")
    parser.add_argument("--image_percent", type=float, default=50.0, help="Target percentage for image datasets.")
    parser.add_argument("--multiimage_percent", type=float, default=5.0, help="Target percentage for multi-image datasets.")
    parser.add_argument("--video_percent", type=float, default=36.0, help="Target percentage for video datasets.")
    parser.add_argument("--short_video_factor", type=float, default=0.5,
                        help="Multiplier for video datasets with an upper duration bound <= 60 seconds (as determined by the _X_Y_s pattern).")
    parser.add_argument("--mammoth_fraction", type=float, default=0.3,
                        help="Fraction of the overall shared dataset to assign to the MammothVL version (the remainder goes to OneVision).")
    args = parser.parse_args()

    data = load_yaml(args.input_yaml)
    updated_data = update_sampling_strategies(
        data,
        text_percent=args.text_percent,
        image_percent=args.image_percent,
        multiimage_percent=args.multiimage_percent,
        video_percent=args.video_percent,
        short_video_factor=args.short_video_factor,
        mammoth_fraction=args.mammoth_fraction
    )
    
    write_yaml(updated_data, args.output_yaml)
    print(f"[INFO] Updated YAML written to {args.output_yaml}")

if __name__ == "__main__":
    main()