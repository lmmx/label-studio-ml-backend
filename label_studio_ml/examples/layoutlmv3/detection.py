import logging
from typing import Any

import numpy as np
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.examples.layoutlmv3.components import (
    LayoutBlock,
    DetectionResult,
    PredictionResult,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["make_predictions", "unnormalize_box", "iob_to_label", "process_image"]

# def detect_image(self, image: Image, backend: LabelStudioMLBase) -> list[LayoutBlock]:
#     # TODO: change this to HuggingFace style
#     labels = torch.tensor([1], dtype=torch.long)
#     encoding = backend.processor(image)
#     with torch.no_grad():
#         logits = backend.model(**encoding, labels=labels).logits
#     predictions = logits.argmax(-1).squeeze().tolist()
#     labels = encoding.labels.squeeze().tolist()
#     label_idx = torch.argmax(predictions).item()
#     score = predictions.flatten().tolist()[label_idx]
#     return LayoutBlock(label_idx=label_idx, score=score)


def make_predictions(
    backend: LabelStudioMLBase,
    images: list[Image],
    task_ids: list[str],
):
    """
    Called in :meth:`LayoutLMv3Classifier.predict()` with first argument as ``self``.
    """
    predictions = []
    layouts = [process_image(image, backend=backend) for image in images]
    for image, layout, task_id in zip(images, layouts, task_ids):
        shape = image.shape[:2]
        detection_results = [
            DetectionResult.from_backend(
                block=block,
                backend=backend,
                shape=shape,
            )
            for block in layout
        ]
        pred = PredictionResult(
            result=detection_results,
            task=task_id,
            # score=score,
        )
        predictions.append(pred)
    return predictions


# These come from Niels Rogge's LayoutLMv2 and -v3 code demos


def unnormalize_box(bbox, width, height) -> tuple[float, float, float, float]:
    return (
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    )


def iob_to_label(label):
    label = label[2:]
    return label if label else "other"


def process_image(image: Image, backend: LabelStudioMLBase) -> list[LayoutBlock]:
    """
    Called in ``make_predictions``.
    """
    width, height = image.size
    encoding = backend.processor(
        image, truncation=True, return_offsets_mapping=True, return_tensors="pt"
    )
    offset_mapping = encoding.pop("offset_mapping")
    outputs = backend.model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
    id2label = {v: k for v, k in enumerate(backend.labels)}
    true_predictions = [
        id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]
    ]
    true_boxes = [
        unnormalize_box(box, width, height)
        for idx, box in enumerate(token_boxes)
        if not is_subword[idx]
    ]
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        block = LayoutBlock(box=box, label=predicted_label, score=prediction)
        predictions.append(block)
    return predictions
