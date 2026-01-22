import json
import datasets

_CITATION = """\
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
"""

_DESCRIPTION = """
HellaSwag: Can a Machine Really Finish Your Sentence? is a new dataset for commonsense NLI.
"""

# ✅ 本地路径：请根据你自己的路径修改
_URLS = {
    "train": "Structure_Pruning_Evaluation/lm-evaluation-harness/lm_eval/tasks/hellaswag/hellaswag/data/hellaswag_train.jsonl",
    "test": "Structure_Pruning_Evaluation/lm-evaluation-harness/lm_eval/tasks/hellaswag/hellaswag/data/hellaswag_test.jsonl",
    "dev": "Structure_Pruning_Evaluation/lm-evaluation-harness/lm_eval/tasks/hellaswag/hellaswag/data/hellaswag_val.jsonl",
}


class Hellaswag(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "ind": datasets.Value("int32"),
                "activity_label": datasets.Value("string"),
                "ctx_a": datasets.Value("string"),
                "ctx_b": datasets.Value("string"),
                "ctx": datasets.Value("string"),
                "endings": datasets.features.Sequence(datasets.Value("string")),
                "source_id": datasets.Value("string"),
                "split": datasets.Value("string"),
                "split_type": datasets.Value("string"),
                "label": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage="https://rowanzellers.com/hellaswag/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # ✅ 直接返回本地路径，无需 download_and_extract
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": _URLS["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": _URLS["test"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": _URLS["dev"]},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, {
                    "ind": int(data["ind"]),
                    "activity_label": data["activity_label"],
                    "ctx_a": data.get("ctx_a", ""),
                    "ctx_b": data.get("ctx_b", ""),
                    "ctx": data["ctx"],
                    "endings": data.get("endings", []),
                    "source_id": data["source_id"],
                    "split": data["split"],
                    "split_type": data["split_type"],
                    "label": str(data.get("label", "")),
                }
