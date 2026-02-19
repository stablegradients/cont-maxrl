from __future__ import annotations

import numpy as np
import datasets as hf_datasets
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


class ContMaxRLDataset(RLHFDataset):
    """
    Backward-compatible dataset wrapper for experiment4 JSONL/Parquet files.

    Differences from the base RLHFDataset:
    - Accepts .jsonl files (verl's base class only accepts .parquet and .json).
      .jsonl and .json are the same format; the base class check is extension-only.
    - Wraps legacy string prompts into a single user-turn chat message.
    """

    def _read_files_and_tokenize(self):
        # Copied from RLHFDataset._read_files_and_tokenize with .jsonl added.
        dataframes = []
        for data_file in self.data_files:
            if data_file.endswith(".parquet"):
                df = hf_datasets.load_dataset("parquet", data_files=data_file)["train"]
            elif data_file.endswith((".json", ".jsonl")):
                df = hf_datasets.load_dataset("json", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            dataframes.append(df)
        self.dataframe = hf_datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = example.get(self.prompt_key)
        if isinstance(prompt, str):
            example[self.prompt_key] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict):
            example[self.prompt_key] = [prompt]
        elif isinstance(prompt, list):
            if prompt and isinstance(prompt[0], str):
                example[self.prompt_key] = [{"role": "user", "content": "\n".join(prompt)}]
        else:
            raise TypeError(
                f"Unsupported prompt type for key '{self.prompt_key}': {type(prompt)!r}"
            )

        return super()._build_messages(example)
