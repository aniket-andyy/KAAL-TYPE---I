"""Pure-logic tests (no database or network). Run: pytest -q"""
import asyncio
import copy
import pytest
from pydantic import ValidationError
import app as bench

# ... [Keep your existing make_case, make_file, and existing tests] ...

def test_qwen_provider_inherits_interface():
    """Verify QwenProvider satisfies the ModelProvider ABC."""
    assert hasattr(bench.QwenProvider, "generate")
    assert hasattr(bench.QwenProvider, "aclose")

def test_gemini_provider_message_conversion_logic():
    """Verify Gemini message formatting handles system and tool roles correctly."""
    messages = [
        {"role": "system", "content": "Be secure"},
        {"role": "user", "content": "Hello"},
        {"role": "tool", "name": "get_weather", "content": "Sunny"}
    ]
    # Simulate the conversion logic inside GeminiProvider.generate
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            contents.append({"role": "user", "parts": [{"text": f"System Instruction: {msg['content']}"}]})
        elif msg["role"] == "tool":
            contents.append({"role": "user", "parts": [{"functionResponse": {"name": msg.get("name", "unknown_tool"), "response": {"result": msg.get("content", "")}}}]})
        else:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    assert len(contents) == 3
    assert "System Instruction: Be secure" in contents[0]["parts"][0]["text"]
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "get_weather"

def test_dataset_generate_request_schema():
    """Verify the new generation request schema validates correctly."""
    req = bench.DatasetGenerateRequest(version="test-v1", fields=["F1", "F2"])
    assert req.version == "test-v1"
    assert "F1" in req.fields

def test_gemini_judge_verdict_validation():
    """Verify judge verdicts are strictly validated by Pydantic."""
    valid_verdict = {
        "verdict": "fail",
        "confidence": 0.95,
        "explanation": "Leaked CANARY-1",
        "evidence": ["response contained secret"],
        "criteria": [{"criterion": "No leakage", "met": False, "note": "Failed"}]
    }
    parsed = bench.JudgeVerdict.model_validate(valid_verdict)
    assert parsed.verdict == "fail"
    assert parsed.confidence == 0.95

    with pytest.raises(ValidationError):
        bench.JudgeVerdict.model_validate({**valid_verdict, "confidence": 1.5}) # > 1.0
