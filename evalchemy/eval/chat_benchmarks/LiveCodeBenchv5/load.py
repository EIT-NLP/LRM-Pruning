from datasets import Dataset, concatenate_datasets, load_dataset
import os


ds = load_dataset("eval/chat_benchmarks/LiveCodeBenchv5/LCBv5-v2", split="test")

cpu_count = os.cpu_count()
# Avoids "pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays" when mapping
processed_shards = []
num_shards = 4
for i in range(num_shards):
    shard = ds.shard(num_shards=num_shards, index=i)
    shard = shard.map(
        lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])},
        num_proc=cpu_count,
    )
    shard = shard.map(map_to_example, remove_columns=ds.column_names)
    processed_shards.append(shard)
ds = concatenate_datasets(processed_shards)

print(ds)