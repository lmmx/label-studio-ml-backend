import logging
from pathlib import Path
from typing import Type

import torch
from label_studio_tools.core.label_config import parse_config
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)

from label_studio_ml.model import LabelStudioMLBase

# TODO break these out further
from label_studio_ml.utils import get_single_tag_keys
from label_studio_ml.examples.layoutlmv3.components import (
    DEVICE,
    MODEL_FILE,
    Custom_Dataset,
    PredictionResult,
)
from label_studio_ml.examples.layoutlmv3.ls_api import get_annotated_dataset
from label_studio_ml.examples.layoutlmv3.detection import make_predictions
from label_studio_ml.examples.layoutlmv3.url_utils import load_image_from_path_or_url

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["LayoutLMv3Classifier"]


class LayoutLMv3Classifier(LabelStudioMLBase):
    control_type: str = "RectangleLabels"
    object_type: str = "Image"
    hf_hub_name: str = "microsoft/layoutlmv3-base"
    hf_model_cls: Type = LayoutLMv3ForTokenClassification
    hf_processor_cls: Type = LayoutLMv3Processor
    detection_source: str = f"${object_type}".lower()
    detection_type: str = control_type.lower()

    def __init__(self, **kwargs):
        super(LayoutLMv3Classifier, self).__init__(**kwargs)
        self.load_config()
        self.processor = self.tokenizer_cls.from_pretrained(self.hf_hub_name)
        if not self.train_output:
            self.labels = self.info["labels"]
            self.reset_model()
            load_repr = "Initialised with"
        else:
            self.load(self.train_output)
            load_repr = "Loaded from train output with"
        logger.info(f"{load_repr} {self.from_name=}, {self.to_name=}, {self.labels=!s}")

    def _load_model(self, name_or_path: str) -> None:
        assert hasattr(self, "labels"), "Loading model requires labels to be set first"
        self.model = self.hf_model_cls.from_pretrained(
            name_or_path,
            num_labels=len(self.labels),
        )
        self.model.to(DEVICE)
        return

    def reset_model(self) -> None:
        return self._load_model(name_or_path=self.hf_hub_name)

    def load(self, train_output: dict) -> None:
        self.labels = train_output["labels"]
        self._load_model(name_or_path=train_output["model_path"])
        self.model.eval()
        self.batch_size = train_output["batch_size"]  # TODO: review use in `predict`
        self.maxlen = train_output["maxlen"]  # TODO: ditto (source: BERT backend)

    def load_config(self):
        if not self.parsed_label_config:
            raise ValueError("The parsed_label_config attribute is not set")
        try:
            self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
                self.parsed_label_config,
                control_type=self.control_type,
                object_type=self.object_type,
            )
        except BaseException:
            logger.error("Couldn't load parsed_label_config", exc_info=True)
        return

    def load_images_from_urls(image_paths_or_urls: list[Path | str]):
        return list(map(load_image_from_path_or_url, image_paths_or_urls))

    def predict(self, tasks, **kwargs) -> list[PredictionResult]:
        # get data for prediction from tasks
        image_urls = [task["data"][self.value] for task in tasks]
        task_ids = [task["id"] for task in tasks]
        images = self.load_images_from_urls(image_urls)
        return make_predictions(backend=self, images=images, task_ids=task_ids)

    def fit(self, completions, workdir=None, **kwargs):
        # check if training is from web hook
        if kwargs.get("data"):
            project_id = kwargs["data"]["project"]["id"]
            tasks = get_annotated_dataset(project_id)
            if not self.parsed_label_config:
                self.parsed_label_config = parse_config(
                    kwargs["data"]["project"]["label_config"]
                )
                self.load_config()
        # ML training without web hook
        else:
            tasks = completions
        # Create training params with batch size = 1 as text are different size
        training_args = TrainingArguments(
            "test_trainer", per_device_train_batch_size=1, per_device_eval_batch_size=1
        )
        # Prepare training data
        input_texts = []
        input_labels = []
        for task in tasks:
            if not task.get("annotations"):
                continue
            input_text = task["data"].get(self.value)
            input_texts.append(
                torch.flatten(self.tokenizer.encode(input_text, return_tensors="pt"))
            )
            annotation = task["annotations"][0]
            output_label = annotation["result"][0]["value"]["choices"][0]
            output_label_idx = self.labels.index(output_label)
            output_label_idx = torch.tensor([[output_label_idx]], dtype=torch.int)
            input_labels.append(output_label_idx)
        print(f"Train dataset length: {len(tasks)}")
        my_dataset = Custom_Dataset((input_texts, input_labels))
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=my_dataset,
            # eval_dataset=small_eval_dataset
        )
        trainer.train()
        self.model.save_pretrained(MODEL_FILE)
        train_output = {"labels": self.labels, "model_file": MODEL_FILE}
        return train_output
