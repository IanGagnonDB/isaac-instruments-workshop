# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt-focused transcript V2
# MAGIC
# MAGIC This optional job consumes the sparse matched-frame manifest produced by the
# MAGIC segmentation job. It gives the vision model the user's original SAM3 prompt,
# MAGIC then asks for a focused temporal narrative. It does not block video delivery.

# COMMAND ----------

dbutils.widgets.text("trigger_location", "")
dbutils.widgets.text("prompt", "")
dbutils.widgets.text("fmapi_model", "databricks-gemini-3-8-flash")
dbutils.widgets.text("volume_root", "")

trigger_location = dbutils.widgets.get("trigger_location").removeprefix("dbfs:")
requested_prompt = dbutils.widgets.get("prompt").strip()
fmapi_model = dbutils.widgets.get("fmapi_model").strip()
volume_root = dbutils.widgets.get("volume_root").rstrip("/")

if not trigger_location or not fmapi_model:
    raise ValueError("trigger_location and fmapi_model are required.")

# COMMAND ----------

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

location = Path(trigger_location)
if location.suffix.lower() == ".json":
    manifest_path = location
else:
    manifest_path = Path(volume_root) / "manifests" / f"{location.stem}.json"
if not manifest_path.is_file():
    raise FileNotFoundError(f"V2 segmentation manifest not found: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
prompt = requested_prompt or str(manifest["prompt"])
if requested_prompt and requested_prompt != manifest["prompt"]:
    raise ValueError(
        "The transcript prompt must match the segmentation prompt in the manifest."
    )

description_path = Path(volume_root) / "descriptions" / f"{location.stem}.txt"
description_path.parent.mkdir(parents=True, exist_ok=True)
frames = manifest.get("transcript_frames", [])

# COMMAND ----------

workspace_host = spark.conf.get("spark.databricks.workspaceUrl")
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = context.apiToken().get()

client = OpenAI(
    api_key=token,
    base_url=f"https://{workspace_host}/serving-endpoints",
)


def text_content(completion):
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    raise TypeError(f"Unsupported model response type: {type(content)!r}")


def frame_content(frame):
    path = Path(frame["path"])
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {
            "type": "text",
            "text": (
                f"Frame at {frame['timestamp_seconds']:.2f} seconds "
                f"(source frame {frame['frame_index']}):"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        },
    ]


def describe_section(section_number, section_frames):
    content = [
        {
            "type": "text",
            "text": (
                f'The video segmentation target supplied by the user is: "{prompt}". '
                "Describe only what these ordered frames show about that target or "
                "closely related activity. Identify changes over time and uncertainty. "
                "Do not invent events between sampled frames. Be succinct and functional."
            ),
        }
    ]
    for frame in section_frames:
        content.extend(frame_content(frame))

    completion = client.chat.completions.create(
        model=fmapi_model,
        messages=[{"role": "user", "content": content}],
    )
    return f"SECTION {section_number}\n{text_content(completion)}"


def create_transcript():
    if not frames:
        return (
            f'No sampled frames matched the segmentation prompt **"{prompt}"** '
            f"at threshold {manifest['threshold']}."
        )

    section_size = 6
    sections = [
        frames[index : index + section_size]
        for index in range(0, len(frames), section_size)
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        descriptions = list(
            executor.map(
                lambda item: describe_section(item[0] + 1, item[1]),
                enumerate(sections),
            )
        )

    synthesis_prompt = f"""
Create a concise Markdown temporal narrative from the observations below.

The user's segmentation target is: "{prompt}".

Focus on when and how that target appears or changes. Preserve timestamps,
distinguish direct visual evidence from uncertainty, and do not add events that
are not present in the observations. Mention that the narrative is based on
sparse matched frames. Return only Markdown.

{chr(10).join(descriptions)}
""".strip()
    completion = client.chat.completions.create(
        model=fmapi_model,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    return text_content(completion)


started = time.perf_counter()
transcript = create_transcript()
description_path.write_text(transcript, encoding="utf-8")
elapsed = round(time.perf_counter() - started, 3)

print(transcript)
dbutils.notebook.exit(
    json.dumps(
        {
            "description_path": str(description_path),
            "frame_count": len(frames),
            "elapsed_seconds": elapsed,
        }
    )
)
