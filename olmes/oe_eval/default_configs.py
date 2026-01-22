## Important: Only change default values with extreme case as it will invalidate model/task hashes!

MODEL_DEFAULTS = {
    "slice_model_path": None, # For loading a part of the model from a different path
    "adapter_path": None,  # For loading adapters from a different path
    "model": None,
    "model_type": "hf",
    "revision": None,
    "trust_remote_code": None,
    "tokenizer": None,
    "max_length": 2048,
    "model_path": None,
    "load_in_4bit": False,
    "load_in_8bit": False,
    "quantization_config": None,  # TODO: Process load_in_4bit and load_in_8bit into this
    "dtype": "auto",
    "use_fast_tokenizer": True,
    "add_bos_token": False,
    "prefix_token_id": None,
    "parallelize": False,
    "device_map_option": "auto",
    "peft": None,
    "delta": None,
    "autogptq": False,
    "chat_model": False,
    "chat_template": None,
    "vllm_for_mc": False,
    "vllm_logit_bias": None,
    "gpu_memory_utilization": None,
    "process_output": None,
    "api_base_url": None,  # Used for litellm models
}

TASK_DEFAULTS = {
    "task_name": None,
    "task_core": None,
    "limit": None,
    "split": None,
    "num_shots": 0,
    "fewshot_seed": 43,
    "primary_metric": None,
    "random_subsample_seed": 43,
    "context_kwargs": None,
    "generation_kwargs": None,
    "metric_kwargs": None,
    "native_id_field": "id",
    "fewshot_source": None,
    "dataset_path": None,
    "dataset_name": None,
    "use_chat_format": None,
    "version": None,
    "chat_overrides": None,
    "revision": None,
    "compute_gold_bpb": False,
    "external_eval": None,
    "skip_model_judges": None,  # Do not include model judges in metrics
    "model_max_length": None,  # To override model max_length
}
