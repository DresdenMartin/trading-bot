import os
import json
from unittest.mock import MagicMock, patch
os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'

import analyze_with_chatgpt


def test_analyze_summary_parses_json():
    # mock OpenAI client
    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content='{"summary":"ok","signals":{},"suggested_action":"hold","confidence":0.5,"recommended_size":null,"rationale":"none"}'))]
    mock_client.chat.completions.create.return_value = resp

    with patch('analyze_with_chatgpt.OpenAI', return_value=mock_client):
        os.environ['OPENAI_API_KEY'] = 'fake'
        out = analyze_with_chatgpt.analyze_summary('ACME', {'price': 100})
        assert isinstance(out, dict)
        assert out.get('summary') == 'ok'
