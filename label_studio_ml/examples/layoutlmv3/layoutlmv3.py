import json
import logging
from pathlib import Path

import requests
import torch
from label_studio_tools.core.label_config import parse_config
from transformers import (
    LayoutLMv3Processor,
    ElectraForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_env

HOSTNAME = get_env("HOSTNAME", "http://localhost:8080")
API_KEY = get_env("API_KEY")

logger.info("=> LABEL STUDIO HOSTNAME = ", HOSTNAME)
if not API_KEY:
    logger.warning("=> WARNING! API_KEY is not set")

MODEL_FILE = "my_model"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LayoutLMv3Classifier(LabelStudioMLBase):
    control_type: str = "RectangleLabels"
    object_type: str = "Image"
    hf_model_name: str = "google/electra-small-discriminator"
    hf_processor_name: str = "microsoft/layoutlmv3-base"
    hf_model_class: type = ElectraForSequenceClassification
    hf_processor_class: type = LayoutLMv3Processor

    def __init__(self, **kwargs):
        super(LayoutLMv3Classifier, self).__init__(**kwargs)
        self.load_config()
        self.processor = self.tokenizer_class.from_pretrained(self.hf_processor_name)
        model_name = MODEL_FILE if Path(MODEL_FILE).exists() else self.hf_model_name
        self.model = self.hf_model_class.from_pretrained(model_name)

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
            logger.error(f"Couldn't load config from parsed_label_config", exc_info=True)

    def predict(self, tasks, **kwargs):
        # get data for prediction from tasks
        final_results = []
        for task in tasks:
            input_texts = ""
            input_text = task["data"].get(self.value)
            if input_text.startswith("http://"):
                input_text = self._get_text_from_s3(input_text)
            input_texts += input_text

            labels = torch.tensor([1], dtype=torch.long)
            # tokenize data
            input_ids = torch.tensor(
                self.tokenizer.encode(input_texts, add_special_tokens=True)
            ).unsqueeze(0)
            # predict label
            predictions = self.model(input_ids, labels=labels).logits
            predictions = torch.softmax(predictions.flatten(), 0)
            label_count = torch.argmax(predictions).item()
            final_results.append(
                {
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "value": {"choices": [self.labels[label_count]]},
                        }
                    ],
                    "task": task["id"],
                    "score": predictions.flatten().tolist()[label_count],
                }
            )
        return final_results

    def fit(self, completions, workdir=None, **kwargs):
        # check if training is from web hook
        if kwargs.get("data"):
            project_id = kwargs["data"]["project"]["id"]
            tasks = self._get_annotated_dataset(project_id)
            if not self.parsed_label_config:
                self.parsed_label_config = parse_config(kwargs["data"]["project"]["label_config"])
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
            if input_text.startswith("http://"):
                input_text = self._get_text_from_s3(input_text)
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

    def _get_annotated_dataset(self, project_id):
        """Just for demo purposes: retrieve annotated data from Label Studio API"""
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'
        response = requests.get(
            download_url, headers={"Authorization": f"Token {API_KEY}"}
        )
        if response.status_code != 200:
            raise Exception(
                f"Can't load task data using {download_url}, "
                f"response status_code = {response.status_code}"
            )
        return json.loads(response.content)

    def _get_text_from_s3(self, url):
        text = requests.get(url)
        return text.text


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[0][index], self.dataset[1][index]
        return {"input_ids": example, "label": target}

    def __len__(self):
        return len(self.dataset)
