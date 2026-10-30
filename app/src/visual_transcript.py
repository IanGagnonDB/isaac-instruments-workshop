import asyncio
import logging
import time
from typing import Any

from databricks.sdk import WorkspaceClient
from openai import AsyncOpenAI

from live_segment import AnnotatedFrame
from telemetry import elapsed_ms, emit


logger = logging.getLogger("cv-video-app.transcript")

SECTION_SIZE = 6


def async_openai_client(workspace: WorkspaceClient) -> AsyncOpenAI:
    headers = workspace.config.authenticate()
    authorization = headers.get("Authorization", "")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise RuntimeError("Databricks authentication did not return a bearer token.")
    return AsyncOpenAI(
        api_key=token,
        base_url=f"{workspace.config.host.rstrip('/')}/serving-endpoints",
    )


def response_text(completion: Any) -> str:
    if not completion.choices:
        raise RuntimeError("Visual transcription endpoint returned no choices.")
    content = completion.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = "".join(
            item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            for item in content
            if (isinstance(item, dict) and item.get("type") == "text")
            or getattr(item, "type", None) == "text"
        )
        if text.strip():
            return text
    raise RuntimeError("Visual transcription endpoint returned no text.")


async def invoke_chat(
    client: AsyncOpenAI,
    endpoint_name: str,
    content: str | list[dict[str, Any]],
    max_tokens: int,
    operation: str,
    section_number: int | None = None,
) -> str:
    request: dict[str, Any] = {
        "model": endpoint_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if endpoint_name.startswith("databricks-gpt-"):
        request["reasoning_effort"] = "low"
    image_count = (
        sum(item.get("type") == "image_url" for item in content)
        if isinstance(content, list)
        else 0
    )
    started = time.perf_counter()
    emit(
        logger,
        "transcript.model_request_started",
        endpoint=endpoint_name,
        operation=operation,
        section_number=section_number,
        image_count=image_count,
        max_tokens=max_tokens,
    )
    try:
        completion = await client.chat.completions.create(**request)
        text = response_text(completion)
    except Exception as error:
        emit(
            logger,
            "transcript.model_request_failed",
            level=logging.ERROR,
            endpoint=endpoint_name,
            operation=operation,
            section_number=section_number,
            image_count=image_count,
            duration_ms=elapsed_ms(started),
            error_type=type(error).__name__,
        )
        raise
    usage = getattr(completion, "usage", None)
    emit(
        logger,
        "transcript.model_request_completed",
        endpoint=endpoint_name,
        operation=operation,
        section_number=section_number,
        image_count=image_count,
        duration_ms=elapsed_ms(started),
        output_characters=len(text),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
    return text


async def describe_section(
    client: AsyncOpenAI,
    endpoint_name: str,
    prompt: str,
    section_number: int,
    frames: list[AnnotatedFrame],
) -> str:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f'The user asked SAM3 to segment: "{prompt}". '
                "These are ordered frames from the annotated segmentation output. "
                "The green overlays and Count label are model annotations, not objects "
                "in the source scene. Describe what the frames directly show about the "
                "target, how it changes over time, and any uncertainty. Be concise. "
                "Do not invent events between frames."
            ),
        }
    ]
    for frame in frames:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"Frame at {frame.timestamp_seconds:.2f} seconds "
                        f"(source frame {frame.frame_index}):"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame.image_b64}"
                    },
                },
            ]
        )
    description = await invoke_chat(
        client,
        endpoint_name,
        content,
        max_tokens=500,
        operation="describe_section",
        section_number=section_number,
    )
    return f"SECTION {section_number}\n{description}"


async def create_visual_transcript(
    workspace: WorkspaceClient,
    endpoint_name: str,
    prompt: str,
    threshold: float,
    frames: tuple[AnnotatedFrame, ...],
) -> str:
    if not frames:
        emit(
            logger,
            "transcript.skipped",
            endpoint=endpoint_name,
            reason="no_matched_frames",
        )
        return (
            f'No sampled frames matched the segmentation prompt **"{prompt}"** '
            f"at threshold {threshold}."
        )

    started = time.perf_counter()
    client = async_openai_client(workspace)
    try:
        sections = [
            list(frames[index : index + SECTION_SIZE])
            for index in range(0, len(frames), SECTION_SIZE)
        ]
        descriptions = await asyncio.gather(
            *(
                describe_section(
                    client,
                    endpoint_name,
                    prompt,
                    section_number,
                    section,
                )
                for section_number, section in enumerate(sections, start=1)
            )
        )

        synthesis_prompt = f"""
Create a concise Markdown temporal narrative from the ordered observations below.

The user's segmentation target is: "{prompt}".

Focus on when and how the target appears or changes. Preserve timestamps,
distinguish direct visual evidence from uncertainty, and do not add events absent
from the observations. State that the narrative is based on {len(frames)} sparse
annotated frames. Return only Markdown.

{chr(10).join(descriptions)}
""".strip()
        transcript = await invoke_chat(
            client,
            endpoint_name,
            synthesis_prompt,
            max_tokens=1600,
            operation="synthesize",
        )
    finally:
        await client.close()
    emit(
        logger,
        "transcript.completed",
        endpoint=endpoint_name,
        frames=len(frames),
        sections=len(sections),
        duration_ms=elapsed_ms(started),
        output_characters=len(transcript),
    )
    return transcript
