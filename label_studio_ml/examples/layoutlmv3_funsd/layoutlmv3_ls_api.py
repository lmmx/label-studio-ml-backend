from __future__ import annotations

import json

import requests

from layoutlmv3_components import HOSTNAME, API_KEY


__all__ = ["get_annotated_dataset"]


def get_annotated_dataset(
    self, project_id, api_key: str = API_KEY, host_name: str = HOSTNAME
) -> dict:
    """Just for demo purposes: retrieve annotated data from Label Studio API"""
    download_url = f'{host_name.rstrip("/")}/api/projects/{project_id}/export'
    response = requests.get(download_url, headers={"Authorization": f"Token {api_key}"})
    if response.status_code != 200:
        raise Exception(
            f"Can't load task data using {download_url}, "
            f"response status_code = {response.status_code}"
        )
    return json.loads(response.content)
