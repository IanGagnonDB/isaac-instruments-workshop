from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import visual_transcript
from live_segment import AnnotatedFrame


@pytest.fixture
def anyio_backend():
    return "asyncio"


def annotated_frames(count: int) -> tuple[AnnotatedFrame, ...]:
    return tuple(
        AnnotatedFrame(
            frame_index=index * 30,
            timestamp_seconds=float(index),
            image_b64="aW1hZ2U=",
        )
        for index in range(count)
    )


def test_empty_model_response_is_rejected():
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
    )

    with pytest.raises(RuntimeError, match="returned no text"):
        visual_transcript.response_text(completion)


@pytest.mark.anyio
async def test_no_matches_returns_explanation_without_calling_endpoint():
    result = await visual_transcript.create_visual_transcript(
        SimpleNamespace(),
        "databricks-gpt-5-6-luna",
        "white truck",
        0.5,
        (),
    )

    assert 'No sampled frames matched the segmentation prompt **"white truck"**' in result


@pytest.mark.anyio
async def test_visual_transcript_batches_frames_in_parallel_then_synthesizes(
    monkeypatch,
):
    client = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        visual_transcript,
        "async_openai_client",
        lambda _workspace: client,
    )
    calls = []

    async def fake_invoke(
        _client,
        endpoint_name,
        content,
        max_tokens,
        operation,
        section_number=None,
    ):
        calls.append(
            (endpoint_name, content, max_tokens, operation, section_number)
        )
        if isinstance(content, list):
            image_count = sum(item["type"] == "image_url" for item in content)
            return f"described {image_count} frames"
        return "**Final temporal narrative**"

    monkeypatch.setattr(visual_transcript, "invoke_chat", fake_invoke)

    result = await visual_transcript.create_visual_transcript(
        SimpleNamespace(),
        "databricks-gpt-5-6-luna",
        "M1A2 Abrams tank",
        0.5,
        annotated_frames(13),
    )

    section_calls = [call for call in calls if isinstance(call[1], list)]
    synthesis_calls = [call for call in calls if isinstance(call[1], str)]
    assert [sum(item["type"] == "image_url" for item in call[1]) for call in section_calls] == [
        6,
        6,
        1,
    ]
    assert len(synthesis_calls) == 1
    assert synthesis_calls[0][2] == 1600
    assert result == "**Final temporal narrative**"
    client.close.assert_awaited_once()
