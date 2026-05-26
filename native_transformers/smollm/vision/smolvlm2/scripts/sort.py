import json
import os
import re
import argparse

def extract_image_key(image_field):
    """
    Extract a key from the "image" field. The field may be:
      - A single string representing one image path.
      - A list of image paths (strings).
      - Potentially None or some other format (in which case we return None).

    For a list of images, we join the basenames of all image paths with an underscore,
    so if ['path/to/img1.jpg', 'path/to/img2.jpg'] is given, we produce "img1.jpg_img2.jpg".
    """
    if isinstance(image_field, list):
        basenames = [
            os.path.basename(img)
            for img in image_field
            if isinstance(img, str) and img.strip() != ""
        ]
        if basenames:
            return "_".join(basenames)
    elif isinstance(image_field, str) and image_field.strip() != "":
        return os.path.basename(image_field)
    return None

def tokenize_id(id_str):
    """
    Tokenize the id string on common delimiters: hyphens, underscores, and periods.
    Return a set of non-empty tokens.

    For example:
        "idefics375k_new.json-284624"
    might become:
        {"idefics375k", "new", "json", "284624"}
    """
    tokens = re.split(r'[-_.]', id_str)
    tokens = [t for t in tokens if t.strip()]
    return set(tokens)

def create_shared_key(entry):
    """
    Create a shared key for an entry with the following priorities:

      1. Use the image key (the joined basenames of images) if available.
         If there's subset info (from 'datasource' or 'source'), prepend it.
            -> "subset_imagebasename" or just "imagebasename"
      2. If no image key is available, check for numeric tokens in the tokenized 'id' field.
         If any exist, use the first one (you could alter this logic as needed).
            -> "subset_number" or just "number"
      3. If no numeric tokens exist, join all tokens from the 'id' field.
            -> "subset_alljoinedtokens" or "alljoinedtokens"
      4. If the entry has no usable information, return None.

    This function is the heart of how we unify or align dataset entries.
    """
    image_key = extract_image_key(entry.get("image"))
    id_str = entry.get("id", "")
    tokens = tokenize_id(id_str)
    numeric_tokens = sorted(t for t in tokens if t.isdigit())

    # Optionally include subset info if available.
    # E.g., "datasource": "coinstruct" or "source": "SVIT_mix_665K_new.json"
    subset = entry.get("datasource") or entry.get("source") or ""

    # 1) Use image key if available
    if image_key:
        return f"{subset}_{image_key}" if subset else image_key

    # 2) If no image key, fallback to numeric tokens from the ID
    if numeric_tokens:
        first_number = numeric_tokens[0]
        return f"{subset}_{first_number}" if subset else first_number

    # 3) If no numeric tokens, fallback to joining all tokens from the ID
    if tokens:
        joined = "_".join(sorted(tokens))
        return f"{subset}_{joined}" if subset else joined

    # 4) If all else fails, return None
    return None

def process_json_file(file_path):
    """
    Load a JSON file (expected to be a list of dictionaries).
    For each entry, compute the shared_id and store it in "shared_id".
    Also print statistics on how many entries received a valid shared_id.
    Returns the processed list of entries (with 'shared_id' included).
    
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_entries = len(data)
    valid_entries = 0

    for entry in data:
        key = create_shared_key(entry)
        entry["shared_id"] = key
        if key:
            valid_entries += 1

    print(f"[INFO] Processed '{file_path}': {total_entries} total entries; "
          f"{valid_entries} assigned a valid shared_id.")
    return data

def write_json_file(data, file_path):
    """
    Write a list of dictionaries into a JSON file with pretty indentation.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Wrote {len(data)} entries to '{file_path}'")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load two JSON files (each a list of dicts). For each entry, create a shared_id "
            "derived from image filename(s), tokenized parts of 'id', and optional subset info. "
            "Writes out updated JSON files including the new 'shared_id' field."
        )
    )
    parser.add_argument(
        "mammoth_file",
        help="Path to the mammoth JSON file."
    )
    parser.add_argument(
        "onevision_file",
        help="Path to the onevision JSON file."
    )
    parser.add_argument(
        "--output_mammoth",
        default="mammoth_aligned.json",
        help="Output JSON file for the mammoth dataset with shared_id."
    )
    parser.add_argument(
        "--output_onevision",
        default="onevision_aligned.json",
        help="Output JSON file for the onevision dataset with shared_id."
    )

    args = parser.parse_args()

    # Process both JSON files.
    mammoth_data = process_json_file(args.mammoth_file)
    onevision_data = process_json_file(args.onevision_file)

    # (Optional) Check for collisions / diagnostics
    # Build a dictionary from shared_id -> list of entries to see if multiple
    # entries share the same ID within each file.
    mammoth_map = {}
    for e in mammoth_data:
        sid = e.get("shared_id")
        mammoth_map.setdefault(sid, []).append(e)

    onevision_map = {}
    for e in onevision_data:
        sid = e.get("shared_id")
        onevision_map.setdefault(sid, []).append(e)

    # Check how many keys are in common between the two sets (not necessarily one-to-one).
    common_keys = (set(mammoth_map.keys()) - {None}) & (set(onevision_map.keys()) - {None})
    print(f"[INFO] Found {len(common_keys)} non-None shared_ids appearing in BOTH datasets.")

    # (Optional) If you want to confirm one-to-one mapping, you could do further checks here.
    # For example, if you need strict pairing, watch out for collisions where more than one
    # entry has the same 'shared_id' in one or both datasets.

    # Write out updated files with the new 'shared_id' field.
    write_json_file(mammoth_data, args.output_mammoth)
    write_json_file(onevision_data, args.output_onevision)

if __name__ == "__main__":
    """
    Example:
        python sort.py \
            data/mammoth.json data/onevision.json \
            --output_mammoth mammoth_out.json \
            --output_onevision onevision_out.json
    """    
    main()