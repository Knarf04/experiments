from datasets import load_dataset

datasets_to_download = [
    # === Leaderboard ===
    # MMLU-Pro (single config)
    ('TIGER-Lab/MMLU-Pro', None),
    # IFEval (single config)
    ('wis-k/instruction-following-eval', None),
    # MuSR (single config)
    ('TAUR-Lab/MuSR', None),
    # GPQA (3 configs used by leaderboard)
    ('Idavidrein/gpqa', 'gpqa_diamond'),
    ('Idavidrein/gpqa', 'gpqa_extended'),
    ('Idavidrein/gpqa', 'gpqa_main'),
    # MATH (7 configs)
    ('DigitalLearningGmbH/MATH-lighteval', 'algebra'),
    ('DigitalLearningGmbH/MATH-lighteval', 'counting_and_probability'),
    ('DigitalLearningGmbH/MATH-lighteval', 'geometry'),
    ('DigitalLearningGmbH/MATH-lighteval', 'intermediate_algebra'),
    ('DigitalLearningGmbH/MATH-lighteval', 'number_theory'),
    ('DigitalLearningGmbH/MATH-lighteval', 'prealgebra'),
    ('DigitalLearningGmbH/MATH-lighteval', 'precalculus'),
    # BBH (27 configs)
    ('SaylorTwift/bbh', 'boolean_expressions'),
    ('SaylorTwift/bbh', 'causal_judgement'),
    ('SaylorTwift/bbh', 'date_understanding'),
    ('SaylorTwift/bbh', 'disambiguation_qa'),
    ('SaylorTwift/bbh', 'dyck_languages'),
    ('SaylorTwift/bbh', 'formal_fallacies'),
    ('SaylorTwift/bbh', 'geometric_shapes'),
    ('SaylorTwift/bbh', 'hyperbaton'),
    ('SaylorTwift/bbh', 'logical_deduction_five_objects'),
    ('SaylorTwift/bbh', 'logical_deduction_seven_objects'),
    ('SaylorTwift/bbh', 'logical_deduction_three_objects'),
    ('SaylorTwift/bbh', 'movie_recommendation'),
    ('SaylorTwift/bbh', 'multistep_arithmetic_two'),
    ('SaylorTwift/bbh', 'navigate'),
    ('SaylorTwift/bbh', 'object_counting'),
    ('SaylorTwift/bbh', 'penguins_in_a_table'),
    ('SaylorTwift/bbh', 'reasoning_about_colored_objects'),
    ('SaylorTwift/bbh', 'ruin_names'),
    ('SaylorTwift/bbh', 'salient_translation_error_detection'),
    ('SaylorTwift/bbh', 'snarks'),
    ('SaylorTwift/bbh', 'sports_understanding'),
    ('SaylorTwift/bbh', 'temporal_sequences'),
    ('SaylorTwift/bbh', 'tracking_shuffled_objects_five_objects'),
    ('SaylorTwift/bbh', 'tracking_shuffled_objects_seven_objects'),
    ('SaylorTwift/bbh', 'tracking_shuffled_objects_three_objects'),
    ('SaylorTwift/bbh', 'web_of_lies'),
    ('SaylorTwift/bbh', 'word_sorting'),

    # === LongBench v1 (34 configs) ===
    ('Xnhyacinth/LongBench', '2wikimqa'),
    ('Xnhyacinth/LongBench', '2wikimqa_e'),
    ('Xnhyacinth/LongBench', 'dureader'),
    ('Xnhyacinth/LongBench', 'gov_report'),
    ('Xnhyacinth/LongBench', 'gov_report_e'),
    ('Xnhyacinth/LongBench', 'hotpotqa'),
    ('Xnhyacinth/LongBench', 'hotpotqa_e'),
    ('Xnhyacinth/LongBench', 'lcc'),
    ('Xnhyacinth/LongBench', 'lcc_e'),
    ('Xnhyacinth/LongBench', 'lsht'),
    ('Xnhyacinth/LongBench', 'multi_news'),
    ('Xnhyacinth/LongBench', 'multi_news_e'),
    ('Xnhyacinth/LongBench', 'multifieldqa_en'),
    ('Xnhyacinth/LongBench', 'multifieldqa_en_e'),
    ('Xnhyacinth/LongBench', 'multifieldqa_zh'),
    ('Xnhyacinth/LongBench', 'musique'),
    ('Xnhyacinth/LongBench', 'narrativeqa'),
    ('Xnhyacinth/LongBench', 'passage_count'),
    ('Xnhyacinth/LongBench', 'passage_count_e'),
    ('Xnhyacinth/LongBench', 'passage_retrieval_en'),
    ('Xnhyacinth/LongBench', 'passage_retrieval_en_e'),
    ('Xnhyacinth/LongBench', 'passage_retrieval_zh'),
    ('Xnhyacinth/LongBench', 'qasper'),
    ('Xnhyacinth/LongBench', 'qasper_e'),
    ('Xnhyacinth/LongBench', 'qmsum'),
    ('Xnhyacinth/LongBench', 'repobench-p'),
    ('Xnhyacinth/LongBench', 'repobench-p_e'),
    ('Xnhyacinth/LongBench', 'samsum'),
    ('Xnhyacinth/LongBench', 'samsum_e'),
    ('Xnhyacinth/LongBench', 'trec'),
    ('Xnhyacinth/LongBench', 'trec_e'),
    ('Xnhyacinth/LongBench', 'triviaqa'),
    ('Xnhyacinth/LongBench', 'triviaqa_e'),
    ('Xnhyacinth/LongBench', 'vcsum'),

    # === Perplexity: WikiText ===
    ('EleutherAI/wikitext_document_level', 'wikitext-2-raw-v1'),

    # === Perplexity: Pile ===
    # NOTE: Pile is huge. These are downloaded separately below with split='test'.


    # === LongBench v2 (20 configs) ===
    ('recursal/longbench-v2', 'academic_multi'),
    ('recursal/longbench-v2', 'academic_single'),
    ('recursal/longbench-v2', 'agent_history_qa'),
    ('recursal/longbench-v2', 'code_repo_qa'),
    ('recursal/longbench-v2', 'detective'),
    ('recursal/longbench-v2', 'dialogue_history_qa'),
    ('recursal/longbench-v2', 'event_ordering'),
    ('recursal/longbench-v2', 'financial_multi'),
    ('recursal/longbench-v2', 'financial_single'),
    ('recursal/longbench-v2', 'government_multi'),
    ('recursal/longbench-v2', 'government_single'),
    ('recursal/longbench-v2', 'graph_reasoning'),
    ('recursal/longbench-v2', 'legal_multi'),
    ('recursal/longbench-v2', 'legal_single'),
    ('recursal/longbench-v2', 'literary'),
    ('recursal/longbench-v2', 'manyshot_learning'),
    ('recursal/longbench-v2', 'multinews'),
    ('recursal/longbench-v2', 'new_language_translation'),
    ('recursal/longbench-v2', 'table_qa'),
    ('recursal/longbench-v2', 'user_guide_qa'),
]

for entry in datasets_to_download:
    ds_path = entry[0]
    ds_name = entry[1] if len(entry) > 1 else None
    label = f"{ds_path}/{ds_name}" if ds_name else ds_path
    print(f"Downloading {label}...")
    try:
        load_dataset(ds_path, ds_name)
        print(f"  Done.")
    except Exception as e:
        print(f"  FAILED: {e}")

# Pile: patch _split_generators to only download the test split (~5GB vs ~800GB)
def download_pile_test_only(config='all'):
    """Download only the test split of EleutherAI/pile by monkey-patching
    _split_generators to skip the massive train/val downloads."""
    import importlib
    import datasets
    from datasets import load_dataset_builder

    builder = load_dataset_builder('EleutherAI/pile', config)

    if config != 'all':
        print(f"  Skipping EleutherAI/pile/{config}: only 'all' has a test split.")
        return

    # Grab the module where _DATA_URLS is defined
    mod = importlib.import_module(builder.__class__.__module__)
    orig_method = builder.__class__._split_generators

    def _test_only_split_generators(self, dl_manager):
        # Only download the test file URLs
        test_urls = {"test": mod._DATA_URLS[self.config.name]["test"]}
        data_dir = dl_manager.download(test_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"files": data_dir["test"]},
            )
        ]

    # Patch, prepare, restore
    builder.__class__._split_generators = _test_only_split_generators
    try:
        builder.download_and_prepare()
    finally:
        builder.__class__._split_generators = orig_method

print("Downloading EleutherAI/pile/all (test only)...")
try:
    download_pile_test_only('all')
    print("  Done.")
except Exception as e:
    print(f"  FAILED: {e}")
