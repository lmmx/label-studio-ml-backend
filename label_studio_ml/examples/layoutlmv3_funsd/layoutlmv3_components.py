from __future__ import annotations

import logging
from typing import TypedDict

import torch

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_env

__all__ = [
    "HOSTNAME",
    "API_KEY",
    "DEVICE",
    "MODEL_FILE",
    "convert_box_to_value",
    "LayoutBlock",
    "BlockValue",
    "DetectionResult",
    "PredictionResult",
    "Custom_Dataset",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HOSTNAME = get_env("HOSTNAME", "http://localhost:8080")
API_KEY = get_env("API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("=> LABEL STUDIO HOSTNAME = ", HOSTNAME)
if not API_KEY:
    logger.warning("=> WARNING! API_KEY is not set")

MODEL_FILE = "my_model"


def convert_box_to_value(box: tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


class LayoutBlock(TypedDict):
    box: tuple[float, float, float, float]
    label: str
    score: float


class BlockValue(TypedDict):
    height: int
    rectanglelabels: list[str]
    rotation: float
    width: int
    x: float
    y: float
    score: float

    @classmethod
    def from_backend(
        cls,
        block: LayoutBlock,
        backend: LabelStudioMLBase,
    ) -> BlockValue:
        box, label, score = block.values()
        x, y, w, h = convert_box_to_value(box)
        return cls(
            height=h,
            rectanglelabels=[label],
            rotation=0,
            width=w,
            x=x,
            y=y,
            score=score,
        )


class DetectionResult(TypedDict):
    from_name: str
    to_name: str
    original_height: int
    original_width: int
    source: str
    type: str
    value: BlockValue

    @classmethod
    def from_backend(
        cls,
        block: LayoutBlock,
        backend: LabelStudioMLBase,
        shape: tuple[int, int],
        # label: str,
        # score: float,
    ) -> BlockValue:
        height, width = shape[:2]
        return cls(
            {
                "from_name": backend.from_name,
                "to_name": backend.to_name,
                "original_height": height,
                "original_width": width,
                "source": backend.detection_source,
                "type": backend.detection_type,
                "value": BlockValue.from_backend(block=block, backend=backend),
            }
        )


class PredictionResult(TypedDict):
    result: list[DetectionResult]
    task: int
    # score: float


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[0][index], self.dataset[1][index]
        return {"input_ids": example, "label": target}

    def __len__(self):
        return len(self.dataset)
