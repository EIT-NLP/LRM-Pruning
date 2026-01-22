# coding=utf-8
import json
import os
import datasets

_CITATION = """\
@inproceedings{Bisk2020,
  author = {Yonatan Bisk and Rowan Zellers and Ronan Le Bras and Jianfeng Gao and Yejin Choi},
  title = {PIQA: Reasoning about Physical Commonsense in Natural Language},
  booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year = {2020},
}
"""

_DESCRIPTION = """\
PIQA focuses on physical commonsense reasoning through multiple choice questions. Given a goal and two solutions, choose the most plausible one.
"""

# ✅ 只需改这里：指向你下载后的本地目录
_LOCAL_DIR = "Structure_Pruning_Evaluation/lm-evaluation-harness/dataset_test/piqa/physicaliqa-train-dev"
_TEST_FILE = "Structure_Pruning_Evaluation/lm-evaluation-harness/dataset_test/piqa/tests.jsonl"

class Piqa(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="plain_text", version=VERSION, description="Plain text"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "goal": datasets.Value("string"),
                "sol1": datasets.Value("string"),
                "sol2": datasets.Value("string"),
                "label": datasets.ClassLabel(names=["0", "1"]),
            }),
            supervised_keys=None,
            homepage="https://yonatanbisk.com/piqa/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "input_filepath": os.path.join(_LOCAL_DIR, "train.jsonl"),
                    "label_filepath": os.path.join(_LOCAL_DIR, "train-labels.lst"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "input_filepath": os.path.join(_LOCAL_DIR, "dev.jsonl"),
                    "label_filepath": os.path.join(_LOCAL_DIR, "dev-labels.lst"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "input_filepath": _TEST_FILE,
                },
            ),
        ]

    def _generate_examples(self, input_filepath, label_filepath=None):
        with open(input_filepath, encoding="utf-8") as input_file:
            inputs = input_file.read().splitlines()

            if label_filepath is not None:
                with open(label_filepath, encoding="utf-8") as label_file:
                    labels = label_file.read().splitlines()
            else:
                labels = [-1] * len(inputs)

            for idx, (row, lab) in enumerate(zip(inputs, labels)):
                data = json.loads(row)
                yield idx, {
                    "goal": data["goal"],
                    "sol1": data["sol1"],
                    "sol2": data["sol2"],
                    "label": lab,
                }
