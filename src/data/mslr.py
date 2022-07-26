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


class MSLR10K(DatasetLoader):
    url = "https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78/root/content"
    zip_file = "MSLR-WEB10K.zip"
    file = "MSLR-WEB10K"
    checksum = "2902142ea33f18c59414f654212de5063033b707d5c3939556124b1120d3a0ba"

    def __init__(
        self, name: str, fold: int, n_features: int, n_results: int, pipeline: Pipeline
    ):
        super().__init__(name, fold, n_features, n_results, pipeline)

    def _parse(self, split: str) -> pd.DataFrame:
        zip_path = download(self.url, self.download_directory / self.zip_file)
        verify_file(zip_path, self.checksum)
        dataset_path = unarchive(zip_path, self.dataset_directory / self.file)

        split = "vali" if split == "val" else split
        path = dataset_path / f"Fold{self.fold}" / f"{split}.txt"

        return read_svmlight_file(path)

    @property
    def folds(self) -> List[int]:
        return [1, 2, 3, 4, 5]

    @property
    def splits(self) -> List[str]:
        return ["train", "test", "val"]


class MSLR30K(DatasetLoader):
    url = "https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbXGPBlwD1rnFdBY/root/content"
    zip_file = "MSLR-WEB30K.zip"
    file = "MSLR-WEB30K"
    checksum = "08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae"

    def __init__(
        self, name: str, fold: int, n_features: int, n_results: int, pipeline: Pipeline
    ):
        super().__init__(name, fold, n_features, n_results, pipeline)

    def _parse(self, split: str) -> pd.DataFrame:
        zip_path = download(self.url, self.download_directory / self.zip_file)
        verify_file(zip_path, self.checksum)
        dataset_path = unarchive(zip_path, self.dataset_directory / self.file)

        split = "vali" if split == "val" else split
        path = dataset_path / f"Fold{self.fold}" / f"{split}.txt"
        return read_svmlight_file(path)

    @property
    def folds(self) -> List[int]:
        return [1, 2, 3, 4, 5]

    @property
    def splits(self) -> List[str]:
        return ["train", "test", "val"]
