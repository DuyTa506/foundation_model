import yaml
import json
import random
from pathlib import Path
import os

def validate_media_paths(json_path, json_data, sample_size=100):
    """
    Validate paths for a sample of elements from the dataset.
    Returns percentage of failed paths.
    """
    base_path = Path(json_path).parent
    sample = random.sample(json_data, min(sample_size, len(json_data)))
    failed_paths = 0
    total_paths = 0
    
    for item in sample:
        paths = []
        # Check for image paths
        if 'image' in item:
            if isinstance(item['image'], list):
                paths.extend(item['image'])
            else:
                paths.append(item['image'])
        
        # Check for video paths
        if 'video' in item:
            if isinstance(item['video'], list):
                paths.extend(item['video'])
            else:
                paths.append(item['video'])
        
        for path in paths:
            total_paths += 1
            full_path = base_path / path
            if not full_path.exists():
                failed_paths += 1
                print(full_path)
    
    return (failed_paths / total_paths * 100) if total_paths > 0 else 0

def process_yaml(yaml_path):
    """
    Process the YAML file, count samples, and validate paths.
    """
    with open(yaml_path, 'r') as file:
        content = file.read()
    
    # Parse YAML content
    data = yaml.safe_load(content)
    
    # Process each dataset
    updated_lines = []
    yaml_lines = content.split('\n')
    current_line = 0
    
    while current_line < len(yaml_lines):
        line = yaml_lines[current_line]
        updated_lines.append(line)
        
        # Check if this line contains json_path and is not a comment
        if 'json_path:' in line and not line.strip().startswith('#'):
            # Check if the next line contains a sample count
            next_line = yaml_lines[current_line + 1] if current_line + 1 < len(yaml_lines) else ""
            if "samples" not in next_line:  # Only process if there's no sample count
                json_path = line.split('json_path:')[1].strip()
                
                try:
                    # Read and process JSON file
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Count samples
                    sample_count = len(json_data)
                    
                    # Validate paths
                    failure_rate = validate_media_paths(json_path, json_data)
                    
                    # Add count and validation info as comment
                    comment = f"    # {sample_count} samples, {failure_rate:.1f}% path validation failures"
                    updated_lines.append(comment)
                    
                    print(f"Processed {json_path}")
                    print(f"Found {sample_count} samples")
                    print(f"Path validation failure rate: {failure_rate:.1f}%")
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"Error processing {json_path}: {str(e)}")
                    updated_lines.append(f"    # Error processing file: {str(e)}")
        
        current_line += 1
    
    # Write updated content back to file
    with open(yaml_path, 'w') as file:
        file.write('\n'.join(updated_lines))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_yaml>")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    process_yaml(yaml_path)