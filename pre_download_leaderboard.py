from datasets import load_dataset

datasets = [
    # MMLU-Pro (no config needed)
    ('TIGER-Lab/MMLU-Pro', None),
    # BBH - each sub-task is a config, just download one to cache the whole dataset
    ('SaylorTwift/bbh', 'boolean_expressions'),
    # GPQA - 3 configs used by leaderboard
    ('Idavidrein/gpqa', 'gpqa_diamond'),
    ('Idavidrein/gpqa', 'gpqa_extended'),
    ('Idavidrein/gpqa', 'gpqa_main'),
    # MATH
    ('DigitalLearningGmbH/MATH-lighteval', 'algebra'),
    # IFEval (no config needed)
    ('wis-k/instruction-following-eval', None),
    # MuSR
    ('TAUR-Lab/MuSR', None),
]

for ds_path, ds_name in datasets:
    label = f"{ds_path}/{ds_name}" if ds_name else ds_path
    print(f"Downloading {label}...")
    load_dataset(ds_path, ds_name)
    print(f"  Done.")
