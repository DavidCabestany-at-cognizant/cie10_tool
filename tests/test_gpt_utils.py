import os
import pytest
from unittest.mock import patch
from ..src.utils.gpt_utils import send_image_to_gpt, load_environment
from PIL import Image

def test_send_image_to_gpt():
    config = load_environment()
    img = Image.new('RGB', (100, 200), color = 'red')
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "response content"}}]
        }
        response = send_image_to_gpt(img, config)
        assert response == "response content"
