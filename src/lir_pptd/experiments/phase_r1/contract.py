from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DatasetModality = Literal['categorical', 'numerical']


@dataclass(frozen=True)
class DatasetContract:
    dataset_id: str
    modality: DatasetModality
    class_count: int | None
    primary_endpoint: str
    secondary_endpoint: str


DATASET_CONTRACTS: dict[str, DatasetContract] = {
    'product': DatasetContract('product', 'categorical', 2, 'one_minus_macro_f1', 'one_minus_accuracy'),
    'duck': DatasetContract('duck', 'categorical', 2, 'one_minus_accuracy', 'one_minus_macro_f1'),
    'dog': DatasetContract('dog', 'categorical', 4, 'one_minus_macro_f1', 'one_minus_accuracy'),
    'weather': DatasetContract('weather', 'numerical', None, 'mae', 'rmse'),
}


def get_dataset_contract(dataset_id: str) -> DatasetContract:
    try:
        return DATASET_CONTRACTS[dataset_id]
    except KeyError as exc:
        raise ValueError(f'unsupported Phase-R1 dataset: {dataset_id}') from exc
