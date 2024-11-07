# Copyright (c) yoomimi

import torch

# Path to the .torch file
# file_path = "datasets/re10k/test/000000.torch"
file_path = "data/mipnerf/test/000000.torch"

# Load the .torch file
data = torch.load(file_path)

# Check if data is a dictionary
if isinstance(data, dict):
    print("skip!!")
# Check if data is a list
elif isinstance(data, list):
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            print(f"Item {idx}: Dictionary with {len(item)} keys")
            for sub_key, sub_value in item.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"  - {sub_key}: {sub_value} with shape {sub_value.shape}")
                    # 첫 번째 행을 출력
                    first_row = sub_value[0]
                    # 생략 없이 전체 출력
                    torch.set_printoptions(threshold=10000)
                    print(f"  - {first_row}")
                elif isinstance(sub_value, list):
                    print(f"  - {sub_key}: List with {len(sub_value)} items")
                    for j, sub_item in enumerate(sub_value):
                        if isinstance(sub_item, torch.Tensor):
                            print(f"    - Item {j}: {sub_item} with shape {sub_item.shape}")
                        else:
                            print(f"    - Item {j}: {type(sub_item)}")
                else:
                    print(f"  - {sub_key}: {type(sub_value)}")
        else:
            print(f"Item {idx}: {type(item)}")
else:
    print("Unsupported data type:", type(data))
