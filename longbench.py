import json
import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader

class LongBenchDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Args:
            dataset_path (str): Path ending in the dataset name.
                                Example: "downloads/data/hotpotqa" 
                                -> Will look for "downloads/data/hotpotqa.jsonl"
        """
        self.data = []
        self._load_data(dataset_path)

    def _load_data(self, path):
        # 1. Infer dataset name and file path based on user requirement
        # "where the last part is the dataset_name"
        path = os.path.normpath(path)
        dataset_name = os.path.basename(path)
        
        # We try to find the .jsonl file. 
        # Most probable: The user points to ".../hotpotqa" but the file is ".../hotpotqa.jsonl"
        potential_files = [
            path + ".jsonl",                     # Case: .../data/hotpotqa -> .../data/hotpotqa.jsonl
            os.path.join(path, f"{dataset_name}.jsonl"), # Case: .../data/hotpotqa/hotpotqa.jsonl
            path                                 # Case: User pointed directly to the .jsonl file
        ]

        target_file = None
        for p in potential_files:
            if os.path.isfile(p):
                target_file = p
                break
        
        if not target_file:
            raise FileNotFoundError(
                f"Could not find a jsonl file for dataset '{dataset_name}'. "
                f"Checked: {potential_files}"
            )

        print(f"Loading {dataset_name} from: {target_file}")

        # 2. Parse the JSONL file
        with open(target_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                
                # Standardizing the output format based on the HF script features
                entry = {
                    "input": item.get("input", ""),
                    "context": item.get("context", ""),
                    "answers": item.get("answers", []),
                    "length": item.get("length", 0),
                    "dataset": item.get("dataset", dataset_name),
                    "language": item.get("language", ""),
                    "all_classes": item.get("all_classes", None),
                    "_id": item.get("_id", "")
                }
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate_fn(batch):
    """
    Custom collate to handle list fields like 'answers' and 'all_classes' 
    which default PyTorch collation might choke on if they vary in size.
    """
    batch_out = {}
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # Keep strings and lists as standard Python lists (don't tensor-ify)
        if isinstance(values[0], (str, list)) or values[0] is None:
            batch_out[key] = values
        else:
            # Convert numbers/integers to tensors
            try:
                batch_out[key] = torch.tensor(values)
            except:
                batch_out[key] = values
                
    return batch_out

if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Load LongBench dataset from a local directory.")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=True, 
        help="Path to the dataset directory (e.g., 'data/hotpotqa')."
    )
    
    args = parser.parse_args()
    
    try:
        # 1. Create Dataset
        dataset = LongBenchDataset(args.dataset_dir)
        
        # 2. Wrap in DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            collate_fn=custom_collate_fn
        )

        # 3. Iterate to verify
        print(f"Dataset size: {len(dataset)}")
        for i, batch in enumerate(dataloader):
            print(f"\n--- Batch {i} ---")
            
            # Print ALL available keys to prove they exist
            print("Available Fields:", list(batch.keys()))
            
            print(f"Dataset Name: {batch['dataset']}")  # Verify your directory logic
            print(f"Inputs:       {batch['input']}")
            print(f"Answers:      {batch['answers']}")
            print(f"All Classes:  {batch['all_classes']}") # Will be None for QA, list for classification
            print(f"IDs:          {batch['_id']}")
            print(f"Context:      {batch['context']}")
            print(f"Length:       {batch['length']}")

            # Stop after 1 batch for demonstration
            break
            
    except FileNotFoundError as e:
        print(f"Error: {e}")