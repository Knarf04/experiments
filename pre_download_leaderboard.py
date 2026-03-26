from datasets import load_dataset
for ds in [
    'TIGER-Lab/MMLU-Pro',
    'SaylorTwift/bbh',
    'Idavidrein/gpqa',
    'DigitalLearningGmbH/MATH-lighteval',
    'wis-k/instruction-following-eval',
    'TAUR-Lab/MuSR',
    'Xnhyacinth/LongBench',
]:
    print(f'Downloading {ds}...')
    load_dataset(ds)
    print(f'  Done.')
