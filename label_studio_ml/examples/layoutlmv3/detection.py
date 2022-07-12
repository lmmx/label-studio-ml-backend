import logging
from typing import Any

import torch

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.examples.layoutlmv3.components import (
    LayoutBlock,
    DetectionResult,
    PredictionResult,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["Image", "detect_image", "make_predictions"]

Image = Any


def detect_image(self, image: Image, backend: LabelStudioMLBase) -> list[LayoutBlock]:
    # TODO: change this to HuggingFace style
    labels = torch.tensor([1], dtype=torch.long)
    encoding = backend.processor(image)
    with torch.no_grad():
        logits = backend.model(**encoding, labels=labels).logits
    predictions = logits.argmax(-1).squeeze().tolist()
    labels = encoding.labels.squeeze().tolist()
    label_idx = torch.argmax(predictions).item()
    score = predictions.flatten().tolist()[label_idx]
    return LayoutBlock(label_idx=label_idx, score=score)


def make_predictions(
    backend: LabelStudioMLBase,
    images: list[Image],
    task_ids: list[str],
):
    predictions = []
    layouts = [detect_image(image, backend=backend) for image in images]
    for image, layout, task_id in zip(images, layouts, task_ids):
        shape = image.shape[:2]
        label_idx, score = layout.values()
        label = backend.labels[label_idx]
        detection_results = [
            DetectionResult.from_backend(
                block=block, backend=backend, shape=shape, label=label, score=score
            )
            for block in layout
        ]
        pred = PredictionResult(
            result=detection_results,
            task=task_id,
            score=score,
        )
        predictions.append(pred)
    return predictions
