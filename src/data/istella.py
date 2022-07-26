from typing import List

import pandas as pd

from src.data.base import DatasetLoader
from src.data.preprocessing import Pipeline
from src.util.file import (
    download,
    unarchive,
    read_svmlight_file,
    verify_file,
)


class Istella(DatasetLoader):
    url = "http://library.istella.it/dataset/istella-s-letor.tar.gz"
    zip_file = "istella-s-letor.tar.gz"
    file = "ISTELLA"
    checksum = "41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981"

    def __init__(self, name: str, fold: int, n_features: int, pipeline: Pipeline):
        super().__init__(name, fold, n_features, pipeline)

    def _parse(self, split: str) -> pd.DataFrame:
        zip_path = download(self.url, self.download_directory / self.zip_file)
        verify_file(zip_path, self.checksum)
        dataset_path = unarchive(zip_path, self.dataset_directory / self.file)

        split = "vali" if split == "val" else split
        path = dataset_path / "sample" / f"{split}.txt"

        return read_svmlight_file(path)

    @property
    def folds(self) -> List[int]:
        return [1]

    @property
    def splits(self) -> List[str]:
        return ["train", "val", "test"]
