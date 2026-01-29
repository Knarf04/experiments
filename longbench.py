from datasets import load_dataset

data_dir = "/gpfs/hshen/datasets/longbench-v1"
data_dir = None
ds = load_dataset("longbench_local.py", name="qasper", data_dir=data_dir, split="test")
first_two = ds[:2]
print(first_two)