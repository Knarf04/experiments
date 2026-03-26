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

    # === Perplexity: WikiText (test split only) ===
    ('EleutherAI/wikitext_document_level', 'wikitext-2-raw-v1', 'test'),

    # === Perplexity: Pile (test split only — full dataset is ~800GB) ===
    ('EleutherAI/pile', 'all', 'test'),
    ('EleutherAI/pile', 'enron_emails', 'test'),
    ('EleutherAI/pile', 'europarl', 'test'),
    ('EleutherAI/pile', 'free_law', 'test'),
    ('EleutherAI/pile', 'hacker_news', 'test'),
    ('EleutherAI/pile', 'nih_exporter', 'test'),
    ('EleutherAI/pile', 'pubmed', 'test'),
    ('EleutherAI/pile', 'pubmed_central', 'test'),
    ('EleutherAI/pile', 'ubuntu_irc', 'test'),
    ('EleutherAI/pile', 'uspto', 'test'),
    ('EleutherAI/pile', 'github', 'test'),

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
    split = entry[2] if len(entry) > 2 else None
    label = f"{ds_path}/{ds_name}" if ds_name else ds_path
    if split:
        label += f" (split={split})"
    print(f"Downloading {label}...")
    try:
        load_dataset(ds_path, ds_name, split=split)
        print(f"  Done.")
    except Exception as e:
        print(f"  FAILED: {e}")
