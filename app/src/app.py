import asyncio
import io
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, TypedDict, TypeVar
from urllib.parse import quote

import anyio
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from live_segment import segment_video
from telemetry import (
    current_trace_id,
    elapsed_ms,
    emit,
    trace_context,
    valid_trace_id,
)
from visual_transcript import create_visual_transcript


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
)
logger = logging.getLogger("cv-video-app")

AUTO_SEGMENT_VOLUME_PATH = os.getenv(
    "AUTO_SEGMENT_VOLUME_PATH", "/Volumes/main/video/auto_segment"
).rstrip("/")
INPUTS_DIR = f"{AUTO_SEGMENT_VOLUME_PATH}/inputs"
OUTPUTS_DIR = f"{AUTO_SEGMENT_VOLUME_PATH}/outputs"
DESCRIPTIONS_DIR = f"{AUTO_SEGMENT_VOLUME_PATH}/descriptions"
EVENTS_TABLE = os.getenv("EVENTS_TABLE", "main.video.edge_events")
REVIEWS_TABLE = os.getenv("REVIEWS_TABLE", "main.video.event_reviews")
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
STATIC_DIR = Path(__file__).parent / "static"

VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}
MAX_UPLOAD_BYTES = 5 * 1024**3
MAX_LIVE_INPUT_BYTES = 512 * 1024**2
DEFAULT_FRAME_STRIDE = 15
DEFAULT_THRESHOLD = 0.5
DEFAULT_OUTPUT_FPS = 8.0
DEFAULT_TRANSCRIPT_MODEL = "luna"
TRANSCRIPT_ENDPOINT_ENV = {
    "luna": "VISUAL_TRANSCRIPT_ENDPOINT",
    "gemini": "GEMINI_VISUAL_TRANSCRIPT_ENDPOINT",
}
T = TypeVar("T")
LIVE_SEGMENT_LOCK = asyncio.Lock()
EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SEVERITY_LEVELS = ("none", "low", "medium", "high", "critical")
MAX_REVIEW_NOTES = 2_000
MAX_LISTED_EVENTS = 200
EVENT_TYPES = (
    "stop_sign_violation",
    "door_opening_hazard",
    "following_too_close",
    "harsh_braking",
    "harsh_acceleration",
    "pedestrian_conflict",
    "unsafe_cut_in",
    "lane_departure",
    "speeding",
    "near_miss",
    "collision",
    "no_event",
    "other",
)
CLIENT_EVENT_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
CLIENT_FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


@lru_cache(maxsize=1)
def get_workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


async def call_sdk(operation: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(operation, *args, **kwargs)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def configured_endpoint_name() -> str:
    endpoint = os.getenv("SAM3_ENDPOINT", "").strip()
    if not endpoint:
        raise HTTPException(503, "The live SAM3 endpoint is not configured.")
    return endpoint


def configured_endpoint_names() -> tuple[str, ...]:
    endpoints = tuple(
        endpoint
        for index in range(1, 5)
        if (endpoint := os.getenv(f"SAM3_ENDPOINT_{index}", "").strip())
    )
    return endpoints or (configured_endpoint_name(),)


def configured_transcript_endpoint_name(model: str) -> str:
    env_name = TRANSCRIPT_ENDPOINT_ENV.get(model)
    if not env_name:
        raise HTTPException(400, "Choose a supported visual summary model.")
    endpoint = os.getenv(env_name, "").strip()
    if not endpoint:
        raise HTTPException(503, f"The {model} visual summary endpoint is not configured.")
    return endpoint


def safe_video_name(filename: str) -> str:
    if not filename or filename != os.path.basename(filename):
        raise HTTPException(400, "A video filename is required.")
    if (
        len(filename.encode("utf-8")) > 255
        or any(character in filename for character in '\0\r\n"')
        or filename in {".", ".."}
        or PurePosixPath(filename).suffix.lower() not in VIDEO_EXTENSIONS
    ):
        raise HTTPException(400, "Choose a supported video file.")
    return filename


def upload_video_name(filename: str) -> str:
    safe = safe_video_name(filename)
    path = PurePosixPath(safe)
    prefix = f"{now_utc().strftime('%Y-%m-%dT%H%M')}-"
    stem_budget = 255 - len(prefix.encode("utf-8")) - len(path.suffix.encode("utf-8"))
    stem_bytes = path.stem.encode("utf-8")[:stem_budget]
    while True:
        try:
            stem = stem_bytes.decode("utf-8")
            break
        except UnicodeDecodeError:
            stem_bytes = stem_bytes[:-1]
    return f"{prefix}{stem}{path.suffix}"


def rooted_path(root: str, filename: str) -> str:
    safe = safe_video_name(filename)
    candidate = PurePosixPath(root) / safe
    if candidate.parent != PurePosixPath(root):
        raise HTTPException(403, "The requested path is outside the configured volume.")
    return candidate.as_posix()


def description_path(filename: str) -> str:
    safe = safe_video_name(filename)
    return (PurePosixPath(DESCRIPTIONS_DIR) / f"{PurePosixPath(safe).stem}.txt").as_posix()


@dataclass(frozen=True)
class RunSubmission:
    source_name: str
    prompt: str
    frame_stride: int = DEFAULT_FRAME_STRIDE
    truncate: bool = True
    threshold: float = DEFAULT_THRESHOLD
    output_fps: float | None = DEFAULT_OUTPUT_FPS
    transcript_model: str = DEFAULT_TRANSCRIPT_MODEL

    @classmethod
    def parse(cls, payload: Any) -> "RunSubmission":
        if not isinstance(payload, dict):
            raise HTTPException(400, "Request body must be a JSON object.")

        source_name = safe_video_name(str(payload.get("source_name", "")).strip())
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise HTTPException(400, "Describe the object to find.")
        if len(prompt) > 1_000:
            raise HTTPException(400, "The prompt must be 1,000 characters or fewer.")

        try:
            frame_stride = int(payload.get("frame_stride", DEFAULT_FRAME_STRIDE))
        except (TypeError, ValueError) as error:
            raise HTTPException(400, "Frame stride must be a positive integer.") from error
        if frame_stride < 1:
            raise HTTPException(400, "Frame stride must be a positive integer.")

        truncate = payload.get("truncate", True)
        if not isinstance(truncate, bool):
            raise HTTPException(400, "Truncate must be true or false.")

        try:
            threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError) as error:
            raise HTTPException(400, "Threshold must be a number from 0.25 to 0.9.") from error
        if not 0.25 <= threshold <= 0.9:
            raise HTTPException(400, "Threshold must be a number from 0.25 to 0.9.")

        raw_output_fps = payload.get("output_fps", DEFAULT_OUTPUT_FPS)
        if raw_output_fps is None:
            output_fps = None
        else:
            try:
                output_fps = float(raw_output_fps)
            except (TypeError, ValueError) as error:
                raise HTTPException(
                    400, "Output FPS must be null or a number from 1 to 30."
                ) from error
            if not 1 <= output_fps <= 30:
                raise HTTPException(
                    400, "Output FPS must be null or a number from 1 to 30."
                )

        transcript_model = str(
            payload.get("transcript_model", DEFAULT_TRANSCRIPT_MODEL)
        ).strip().lower()
        if transcript_model not in TRANSCRIPT_ENDPOINT_ENV:
            raise HTTPException(400, "Choose Luna or Gemini for the visual summary.")

        return cls(
            source_name=source_name,
            prompt=prompt,
            frame_stride=frame_stride,
            truncate=truncate,
            threshold=threshold,
            output_fps=output_fps,
            transcript_model=transcript_model,
        )


async def metadata_exists(path: str) -> bool:
    try:
        await call_sdk(get_workspace_client().files.get_metadata, path)
        return True
    except NotFound:
        return False


async def ensure_directory(path: str) -> None:
    try:
        await call_sdk(get_workspace_client().files.get_metadata, path)
    except NotFound:
        await call_sdk(get_workspace_client().files.create_directory, path)


def list_directory(root: str) -> list[Any]:
    return list(
        get_workspace_client().files.list_directory_contents(directory_path=root)
    )


class VideoFile(TypedDict):
    name: str
    size_bytes: int | None


async def list_video_files(root: str, query: str = "") -> list[VideoFile]:
    try:
        items = await call_sdk(list_directory, root)
    except NotFound:
        return []

    query_lower = query.strip().lower()
    videos = {
        entry.name: {
            "name": entry.name,
            "size_bytes": getattr(entry, "file_size", None),
        }
        for entry in items
        if entry.name
        and not entry.is_directory
        and PurePosixPath(entry.name).suffix.lower() in VIDEO_EXTENSIONS
        and (not query_lower or query_lower in entry.name.lower())
    }
    return [videos[name] for name in sorted(videos, key=str.lower)[:50]]


def upload_local_file(local_path: str, destination: str) -> None:
    with open(local_path, "rb") as contents:
        get_workspace_client().files.upload(destination, contents=contents, overwrite=True)


def upload_text(text: str, destination: str) -> None:
    get_workspace_client().files.upload(
        destination,
        contents=io.BytesIO(text.encode("utf-8")),
        overwrite=True,
    )


def download_to_local_file(source: str, destination: str) -> int:
    download = get_workspace_client().files.download(source)
    size = 0
    try:
        with open(destination, "wb") as output:
            while chunk := download.contents.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_LIVE_INPUT_BYTES:
                    raise HTTPException(
                        413,
                        "Live processing supports videos up to 512 MiB.",
                    )
                output.write(chunk)
    finally:
        download.contents.close()
    return size


def open_volume_download(path: str) -> tuple[BinaryIO, int | None]:
    download = get_workspace_client().files.download(path)
    return download.contents, getattr(download, "content_length", None)


def stream_and_close(contents: BinaryIO):
    try:
        while chunk := contents.read(1024 * 1024):
            yield chunk
    finally:
        contents.close()


async def download_text(path: str) -> str | None:
    try:
        download = await call_sdk(get_workspace_client().files.download, path)
    except NotFound:
        return None

    def read_contents() -> str:
        try:
            return download.contents.read().decode("utf-8", errors="replace")
        finally:
            download.contents.close()

    return await call_sdk(read_contents)


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def client_telemetry(request: Request) -> Response:
    body = await request.body()
    if len(body) > 8_192:
        raise HTTPException(413, "Telemetry payload is too large.")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HTTPException(400, "Telemetry payload must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise HTTPException(400, "Telemetry payload must be a JSON object.")

    event = str(payload.get("event", "")).strip()
    if not CLIENT_EVENT_PATTERN.fullmatch(event):
        raise HTTPException(400, "Telemetry event name is invalid.")
    raw_fields = payload.get("fields", {})
    if not isinstance(raw_fields, dict) or len(raw_fields) > 20:
        raise HTTPException(400, "Telemetry fields are invalid.")

    fields: dict[str, str | int | float | bool] = {}
    for key, value in raw_fields.items():
        if not isinstance(key, str) or not CLIENT_FIELD_PATTERN.fullmatch(key):
            raise HTTPException(400, "Telemetry field name is invalid.")
        if isinstance(value, bool | int | float):
            fields[key] = value
        elif isinstance(value, str):
            fields[key] = value[:200]
        elif value is not None:
            raise HTTPException(400, "Telemetry field value is invalid.")
    emit(logger, f"client.{event}", source="browser", client_fields=fields)
    return Response(status_code=204)


async def input_hints(request: Request) -> JSONResponse:
    files = await list_video_files(INPUTS_DIR, request.query_params.get("q", ""))
    return JSONResponse({"items": files, "root": INPUTS_DIR})


async def output_hints(request: Request) -> JSONResponse:
    files = await list_video_files(OUTPUTS_DIR, request.query_params.get("q", ""))
    return JSONResponse({"items": files, "root": OUTPUTS_DIR})


async def upload_input(request: Request) -> JSONResponse:
    started = time.perf_counter()
    filename = upload_video_name(request.path_params["filename"])
    destination = rooted_path(INPUTS_DIR, filename)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Uploads cannot exceed 5 GiB.")
        except ValueError as error:
            raise HTTPException(400, "Content-Length must be an integer.") from error

    await ensure_directory(INPUTS_DIR)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix="cv-upload-")
    os.close(file_descriptor)
    size = 0
    try:
        async with await anyio.open_file(temporary_path, "wb") as temporary_file:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Uploads cannot exceed 5 GiB.")
                await temporary_file.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        await call_sdk(upload_local_file, temporary_path, destination)
    finally:
        await call_sdk(os.unlink, temporary_path)

    emit(
        logger,
        "upload.completed",
        filename=filename,
        bytes=size,
        duration_ms=elapsed_ms(started),
    )
    return JSONResponse(
        {"source_name": filename, "size_bytes": size},
        status_code=201,
    )


async def perform_live_segmentation(
    submission: RunSubmission,
    preview_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    input_path = rooted_path(INPUTS_DIR, submission.source_name)
    run_started = time.perf_counter()
    emit(
        logger,
        "run.started",
        source_name=submission.source_name,
        prompt_characters=len(submission.prompt),
        frame_stride=submission.frame_stride,
        threshold=submission.threshold,
        truncate=submission.truncate,
        output_fps=submission.output_fps,
        transcript_model=submission.transcript_model,
        delivery_mode="preview" if preview_callback else "final",
    )
    if not await metadata_exists(input_path):
        emit(logger, "run.rejected", level=logging.WARNING, reason="input_not_found")
        raise HTTPException(404, "The selected input video was not found.")

    if LIVE_SEGMENT_LOCK.locked():
        emit(logger, "run.rejected", level=logging.WARNING, reason="segmenter_busy")
        raise HTTPException(429, "A live segmentation is already running. Try again shortly.")

    started = now_utc()
    suffix = PurePosixPath(submission.source_name).suffix
    async with LIVE_SEGMENT_LOCK:
        with tempfile.TemporaryDirectory(prefix="cv-live-") as temporary:
            local_input = str(Path(temporary) / f"input{suffix}")
            local_output = str(Path(temporary) / f"output{suffix}")
            download_started = time.perf_counter()
            emit(logger, "volume.input_download_started")
            input_bytes = await call_sdk(
                download_to_local_file,
                input_path,
                local_input,
            )
            emit(
                logger,
                "volume.input_download_completed",
                bytes=input_bytes,
                duration_ms=elapsed_ms(download_started),
            )
            segmentation_started = time.perf_counter()
            try:
                endpoint_names = configured_endpoint_names()
                result = await call_sdk(
                    segment_video,
                    get_workspace_client(),
                    endpoint_names if len(endpoint_names) > 1 else endpoint_names[0],
                    Path(local_input),
                    Path(local_output),
                    prompt=submission.prompt,
                    frame_stride=submission.frame_stride,
                    threshold=submission.threshold,
                    truncate=submission.truncate,
                    output_fps=submission.output_fps,
                    preview_callback=preview_callback,
                )
            except ValueError as error:
                emit(
                    logger,
                    "run.segmentation_failed",
                    level=logging.WARNING,
                    duration_ms=elapsed_ms(segmentation_started),
                    error_type=type(error).__name__,
                )
                raise HTTPException(422, str(error)) from error
            except Exception as error:
                emit(
                    logger,
                    "run.segmentation_failed",
                    level=logging.ERROR,
                    duration_ms=elapsed_ms(segmentation_started),
                    error_type=type(error).__name__,
                )
                raise
            emit(
                logger,
                "run.segmentation_completed",
                duration_ms=elapsed_ms(segmentation_started),
                sampled_frames=result.sampled_frame_count,
                matched_frames=result.matched_frame_count,
            )
            await ensure_directory(OUTPUTS_DIR)
            output_upload_started = time.perf_counter()
            output_bytes = (
                Path(local_output).stat().st_size
                if Path(local_output).exists()
                else None
            )
            emit(
                logger,
                "volume.output_upload_started",
                bytes=output_bytes,
            )
            await call_sdk(
                upload_local_file,
                local_output,
                rooted_path(OUTPUTS_DIR, submission.source_name),
            )
            emit(
                logger,
                "volume.output_upload_completed",
                bytes=output_bytes,
                duration_ms=elapsed_ms(output_upload_started),
            )

    if preview_callback:
        preview_callback(
            {
                "type": "status",
                "stage": "transcribing",
                "message": (
                    f"Describing {len(result.transcript_frames)} annotated frames."
                ),
            }
        )
    description = None
    description_error = None
    transcript_started = time.perf_counter()
    try:
        transcript_endpoint = configured_transcript_endpoint_name(
            submission.transcript_model
        )
        emit(
            logger,
            "run.transcript_started",
            endpoint=transcript_endpoint,
            model=submission.transcript_model,
            annotated_frames=len(result.transcript_frames),
        )
        description = await create_visual_transcript(
            get_workspace_client(),
            transcript_endpoint,
            submission.prompt,
            submission.threshold,
            result.transcript_frames,
        )
        await ensure_directory(DESCRIPTIONS_DIR)
        await call_sdk(
            upload_text,
            description,
            description_path(submission.source_name),
        )
        emit(
            logger,
            "run.transcript_completed",
            annotated_frames=len(result.transcript_frames),
            output_characters=len(description),
            duration_ms=elapsed_ms(transcript_started),
        )
    except Exception as error:
        emit(
            logger,
            "run.transcript_failed",
            level=logging.ERROR,
            annotated_frames=len(result.transcript_frames),
            duration_ms=elapsed_ms(transcript_started),
            error_type=type(error).__name__,
        )
        logger.exception(
            "visual_transcript_failed source=%s",
            submission.source_name,
            exc_info=error,
        )
        description_error = (
            "The segmented video completed, but its visual description failed."
        )

    elapsed = max(1, round((now_utc() - started).total_seconds()))
    emit(
        logger,
        "run.completed",
        source_name=submission.source_name,
        duration_ms=elapsed_ms(run_started),
        elapsed_seconds=elapsed,
        sampled_frames=result.sampled_frame_count,
        matched_frames=result.matched_frame_count,
        transcript_succeeded=description is not None,
    )
    return {
        "output_name": submission.source_name,
        "media_url": f"/api/media/{quote(submission.source_name)}",
        "description": description,
        "description_error": description_error,
        "elapsed_seconds": elapsed,
        "sampled_frame_count": result.sampled_frame_count,
        "matched_frame_count": result.matched_frame_count,
    }


async def create_run(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as error:
        raise HTTPException(400, "Request body must contain valid JSON.") from error
    return JSONResponse(
        await perform_live_segmentation(RunSubmission.parse(payload))
    )


async def stream_run(websocket: WebSocket) -> None:
    trace_id = valid_trace_id(websocket.query_params.get("trace_id"))
    with trace_context(trace_id):
        await _stream_run(websocket)


async def _stream_run(websocket: WebSocket) -> None:
    started = time.perf_counter()
    await websocket.accept()
    emit(logger, "websocket.connected", path=websocket.url.path)
    connected = True
    try:
        submission = RunSubmission.parse(await websocket.receive_json())
        emit(
            logger,
            "websocket.run_received",
            source_name=submission.source_name,
            frame_stride=submission.frame_stride,
            threshold=submission.threshold,
            truncate=submission.truncate,
            output_fps=submission.output_fps,
            transcript_model=submission.transcript_model,
        )
        loop = asyncio.get_running_loop()
        previews: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def publish_preview(event: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(previews.put_nowait, event)

        processing = asyncio.create_task(
            perform_live_segmentation(submission, publish_preview)
        )
        while not processing.done() or not previews.empty():
            try:
                preview = await asyncio.wait_for(previews.get(), timeout=0.1)
            except TimeoutError:
                continue
            if connected:
                try:
                    await websocket.send_json(preview)
                except (RuntimeError, WebSocketDisconnect):
                    connected = False

        output = await processing
        if connected:
            await websocket.send_json({"type": "complete", "output": output})
        emit(
            logger,
            "websocket.run_completed",
            connected=connected,
            duration_ms=elapsed_ms(started),
        )
    except WebSocketDisconnect:
        connected = False
        emit(
            logger,
            "websocket.disconnected",
            level=logging.WARNING,
            duration_ms=elapsed_ms(started),
        )
    except HTTPException as error:
        emit(
            logger,
            "websocket.request_failed",
            level=logging.WARNING,
            status_code=error.status_code,
            duration_ms=elapsed_ms(started),
        )
        if connected:
            await websocket.send_json(
                {"type": "error", "message": str(error.detail)}
            )
    except Exception as error:
        emit(
            logger,
            "websocket.failed",
            level=logging.ERROR,
            error_type=type(error).__name__,
            duration_ms=elapsed_ms(started),
        )
        logger.exception("streaming_segment_error", exc_info=error)
        if connected:
            await websocket.send_json(
                {"type": "error", "message": "Live preview could not be completed."}
            )
    finally:
        if connected:
            await websocket.close()
        emit(
            logger,
            "websocket.closed",
            connected_at_close=connected,
            duration_ms=elapsed_ms(started),
        )


async def output_detail(request: Request) -> JSONResponse:
    filename = safe_video_name(request.path_params["filename"])
    if not await metadata_exists(rooted_path(OUTPUTS_DIR, filename)):
        raise HTTPException(404, "The output video was not found.")
    return JSONResponse(
        {
            "output_name": filename,
            "media_url": f"/api/media/{quote(filename)}",
            "description": await download_text(description_path(filename)),
        }
    )


async def output_media(request: Request) -> StreamingResponse:
    filename = safe_video_name(request.path_params["filename"])
    path = rooted_path(OUTPUTS_DIR, filename)
    try:
        contents, content_length = await call_sdk(open_volume_download, path)
    except NotFound as error:
        raise HTTPException(404, "The output video was not found.") from error

    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": f'inline; filename="{filename}"',
    }
    if content_length:
        headers["Content-Length"] = str(content_length)
    media_type, _ = mimetypes.guess_type(filename)
    return StreamingResponse(
        stream_and_close(contents),
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


def safe_event_id(event_id: str) -> str:
    if not EVENT_ID_PATTERN.fullmatch(event_id or ""):
        raise HTTPException(400, "A valid event ID is required.")
    return event_id


def reviewer_identity(request: Request) -> str:
    # Databricks Apps forwards the signed-in user's identity on every request.
    for header in ("x-forwarded-email", "x-forwarded-preferred-username", "x-forwarded-user"):
        if value := request.headers.get(header, "").strip():
            return value[:254]
    return "unknown"


def quoted_table(name: str) -> str:
    if not TABLE_NAME_PATTERN.fullmatch(name):
        raise HTTPException(500, "The review tables are not configured.")
    return ".".join(f"`{part}`" for part in name.split("."))


def warehouse_id() -> str:
    value = os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip()
    if not value:
        raise HTTPException(503, "The SQL warehouse is not configured.")
    return value


def run_sql(statement: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run one parameterized statement on the app's warehouse and return rows as dicts."""
    from databricks.sdk.service.sql import StatementParameterListItem, StatementState

    client = get_workspace_client()
    response = client.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id(),
        parameters=[
            StatementParameterListItem(
                name=name,
                value=None if value is None else str(value),
                type=(
                    "BOOLEAN"
                    if isinstance(value, bool)
                    else "INT" if isinstance(value, int) else "STRING"
                ),
            )
            for name, value in (parameters or {}).items()
        ],
        wait_timeout="30s",
    )
    # A cold serverless warehouse can take longer than the synchronous wait.
    deadline = time.monotonic() + 120
    while response.status and response.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        if time.monotonic() > deadline:
            raise HTTPException(504, "The SQL warehouse did not respond in time.")
        time.sleep(1)
        response = client.statement_execution.get_statement(response.statement_id)
    if not response.status or response.status.state != StatementState.SUCCEEDED:
        message = response.status.error.message if response.status and response.status.error else ""
        logger.error("sql_statement_failed %s", message)
        raise HTTPException(502, "The review tables could not be queried.")
    if not response.manifest or not response.result or not response.result.data_array:
        return []
    columns = [column.name for column in response.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in response.result.data_array]


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return str(value).lower() == "true"


def as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def review_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "severity": row.get("severity"),
        "edge_classification_correct": as_bool(row.get("edge_classification_correct")),
        "corrected_event_type": row.get("corrected_event_type"),
        "notes": row.get("notes") or "",
        "sam3_prompt": row.get("sam3_prompt"),
        "sam3_matched_frames": as_int(row.get("sam3_matched_frames")),
        "sam3_sampled_frames": as_int(row.get("sam3_sampled_frames")),
        "reviewer": row.get("reviewer"),
        "submitted_at": row.get("submitted_at"),
    }


async def event_list(_: Request) -> JSONResponse:
    rows = await call_sdk(
        run_sql,
        f"""
        SELECT event_id, event_type, event_label, clip, vehicle_id,
               date_format(recorded_at, "yyyy-MM-dd'T'HH:mm:ss'Z'") AS recorded_at,
               edge_severity, review_status, reviewed_severity, edge_classification_correct,
               reviewed_by
        FROM {quoted_table(EVENTS_TABLE)}
        ORDER BY review_status = 'reviewed', recorded_at DESC
        LIMIT {MAX_LISTED_EVENTS}
        """,
    )
    items = [
        {
            **row,
            "review_status": "reviewed" if row.get("review_status") == "reviewed" else "pending",
            "edge_classification_correct": as_bool(row.get("edge_classification_correct")),
        }
        for row in rows
    ]
    return JSONResponse({"items": items, "table": EVENTS_TABLE})


async def event_detail(request: Request) -> JSONResponse:
    event_id = safe_event_id(request.path_params["event_id"])
    rows = await call_sdk(
        run_sql,
        f"SELECT payload, review_status FROM {quoted_table(EVENTS_TABLE)} WHERE event_id = :event_id",
        {"event_id": event_id},
    )
    if not rows:
        raise HTTPException(404, "The event was not found.")
    try:
        event = json.loads(rows[0].get("payload") or "{}")
    except json.JSONDecodeError as error:
        raise HTTPException(500, "The event payload is not valid JSON.") from error
    clip = safe_video_name(str(event.get("clip", "")))
    has_output = await metadata_exists(rooted_path(OUTPUTS_DIR, clip))
    reviews = await call_sdk(
        run_sql,
        f"""
        SELECT event_id, severity, edge_classification_correct, corrected_event_type, notes,
               sam3_prompt, sam3_matched_frames, sam3_sampled_frames, reviewer,
               date_format(submitted_at, "yyyy-MM-dd'T'HH:mm:ss'Z'") AS submitted_at
        FROM {quoted_table(REVIEWS_TABLE)}
        WHERE event_id = :event_id
        ORDER BY submitted_at
        """,
        {"event_id": event_id},
    )
    return JSONResponse(
        {
            "event": event,
            "review_status": "reviewed" if rows[0].get("review_status") == "reviewed" else "pending",
            "clip_url": f"/api/source-media/{quote(clip)}",
            "output_url": f"/api/media/{quote(clip)}" if has_output else None,
            "reviews": [review_record(row) for row in reviews],
            "event_types": list(EVENT_TYPES),
        }
    )


async def submit_review(request: Request) -> JSONResponse:
    event_id = safe_event_id(request.path_params["event_id"])
    try:
        payload = await request.json()
    except Exception as error:
        raise HTTPException(400, "Request body must contain valid JSON.") from error
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")

    severity = str(payload.get("severity", "")).strip().lower()
    if severity not in SEVERITY_LEVELS:
        raise HTTPException(400, f"Severity must be one of: {', '.join(SEVERITY_LEVELS)}.")
    edge_correct = payload.get("edge_classification_correct")
    if not isinstance(edge_correct, bool):
        raise HTTPException(400, "Say whether the edge classification was correct.")
    notes = str(payload.get("notes", "")).strip()
    if len(notes) > MAX_REVIEW_NOTES:
        raise HTTPException(400, f"Notes must be {MAX_REVIEW_NOTES:,} characters or fewer.")
    corrected = str(payload.get("corrected_event_type") or "").strip().lower() or None
    if edge_correct:
        corrected = None
    else:
        if corrected not in EVENT_TYPES:
            raise HTTPException(400, "Choose what the event actually was.")
        if not notes:
            raise HTTPException(400, "Add a note explaining the misclassification.")
    prompt = str(payload.get("sam3_prompt", "")).strip()[:200] or None
    matched = as_int(payload.get("sam3_matched_frames"))
    sampled = as_int(payload.get("sam3_sampled_frames"))

    events = await call_sdk(
        run_sql,
        f"SELECT event_type, edge_severity FROM {quoted_table(EVENTS_TABLE)} WHERE event_id = :event_id",
        {"event_id": event_id},
    )
    if not events:
        raise HTTPException(404, "The event was not found.")

    reviewer = reviewer_identity(request)
    submitted = now_utc().isoformat().replace("+00:00", "Z")
    values = {
        "review_id": f"{event_id}-{submitted}",
        "event_id": event_id,
        "edge_event_type": events[0].get("event_type"),
        "edge_severity": events[0].get("edge_severity"),
        "severity": severity,
        "edge_correct": edge_correct,
        "corrected": corrected,
        "notes": notes,
        "prompt": prompt,
        "matched": matched,
        "sampled": sampled,
        "reviewer": reviewer,
        "submitted": submitted,
    }
    await call_sdk(
        run_sql,
        f"""
        INSERT INTO {quoted_table(REVIEWS_TABLE)} (
          review_id, event_id, edge_event_type, edge_severity, severity,
          edge_classification_correct, corrected_event_type, notes, sam3_prompt,
          sam3_matched_frames, sam3_sampled_frames, reviewer, submitted_at
        ) VALUES (
          :review_id, :event_id, :edge_event_type, :edge_severity, :severity,
          :edge_correct, :corrected, :notes, :prompt,
          :matched, :sampled, :reviewer, to_timestamp(:submitted)
        )
        """,
        values,
    )
    await call_sdk(
        run_sql,
        f"""
        UPDATE {quoted_table(EVENTS_TABLE)}
        SET review_status = 'reviewed', reviewed_severity = :severity,
            edge_classification_correct = :edge_correct, corrected_event_type = :corrected,
            reviewed_by = :reviewer, reviewed_at = to_timestamp(:submitted)
        WHERE event_id = :event_id
        """,
        {
            "severity": severity,
            "edge_correct": edge_correct,
            "corrected": corrected,
            "reviewer": reviewer,
            "submitted": submitted,
            "event_id": event_id,
        },
    )
    return JSONResponse(
        {
            "event_id": event_id,
            "severity": severity,
            "edge_classification_correct": edge_correct,
            "corrected_event_type": corrected,
            "notes": notes,
            "sam3_prompt": prompt,
            "sam3_matched_frames": matched,
            "sam3_sampled_frames": sampled,
            "reviewer": reviewer,
            "submitted_at": submitted,
        },
        status_code=201,
    )


async def source_media(request: Request) -> StreamingResponse:
    filename = safe_video_name(request.path_params["filename"])
    path = rooted_path(INPUTS_DIR, filename)
    try:
        contents, content_length = await call_sdk(open_volume_download, path)
    except NotFound as error:
        raise HTTPException(404, "The source video was not found.") from error

    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": f'inline; filename="{filename}"',
    }
    if content_length:
        headers["Content-Length"] = str(content_length)
    media_type, _ = mimetypes.guess_type(filename)
    return StreamingResponse(
        stream_and_close(contents),
        media_type=media_type or "application/octet-stream",
        headers=headers,
    )


async def spa(request: Request) -> Response:
    path = request.path_params.get("path", "")
    if path.startswith("api/"):
        raise HTTPException(404, "API endpoint not found.")

    index_path = STATIC_DIR / "index.html"
    requested_path = (STATIC_DIR / path).resolve()
    if (
        path
        and requested_path.is_relative_to(STATIC_DIR.resolve())
        and requested_path.is_file()
    ):
        return FileResponse(requested_path)
    if PurePosixPath(path).suffix:
        raise HTTPException(404, "Static asset not found.")
    if index_path.is_file():
        return FileResponse(index_path, headers={"Cache-Control": "no-cache"})
    return JSONResponse(
        {"error": {"message": "Frontend assets have not been built."}},
        status_code=503,
    )


async def http_exception_handler(_: Request, error: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": str(error.detail)}},
        status_code=error.status_code,
        headers=error.headers,
    )


async def unexpected_exception_handler(_: Request, error: Exception) -> JSONResponse:
    emit(
        logger,
        "http.unhandled_exception",
        level=logging.ERROR,
        error_type=type(error).__name__,
    )
    logger.exception("unhandled_request_error", exc_info=error)
    return JSONResponse(
        {"error": {"message": "The request could not be completed."}},
        status_code=500,
    )


async def http_telemetry(request: Request, call_next: Callable[[Request], Any]) -> Response:
    trace_id = valid_trace_id(request.headers.get("x-trace-id"))
    with trace_context(trace_id):
        started = time.perf_counter()
        emit(
            logger,
            "http.request_started",
            method=request.method,
            path=request.url.path,
            content_length=request.headers.get("content-length"),
        )
        try:
            response = await call_next(request)
        except Exception as error:
            emit(
                logger,
                "http.request_failed",
                level=logging.ERROR,
                method=request.method,
                path=request.url.path,
                duration_ms=elapsed_ms(started),
                error_type=type(error).__name__,
            )
            raise
        response.headers["X-Trace-ID"] = current_trace_id()
        emit(
            logger,
            "http.request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_ms(started),
        )
        return response


routes = [
    Route("/api/health", health),
    Route("/api/telemetry", client_telemetry, methods=["POST"]),
    Route("/api/inputs", input_hints),
    Route("/api/inputs/{filename:path}", upload_input, methods=["PUT"]),
    Route("/api/outputs", output_hints),
    Route("/api/outputs/{filename:path}", output_detail),
    Route("/api/runs", create_run, methods=["POST"]),
    WebSocketRoute("/api/segment-stream", stream_run),
    Route("/api/media/{filename:path}", output_media),
    Route("/api/source-media/{filename:path}", source_media),
    Route("/api/events", event_list),
    Route("/api/events/{event_id}", event_detail),
    Route("/api/events/{event_id}/reviews", submit_review, methods=["POST"]),
    Mount("/assets", StaticFiles(directory=STATIC_DIR / "assets", check_dir=False)),
    Route("/{path:path}", spa),
]

app = Starlette(
    debug=False,
    routes=routes,
    middleware=[Middleware(BaseHTTPMiddleware, dispatch=http_telemetry)],
    exception_handlers={
        HTTPException: http_exception_handler,
        Exception: unexpected_exception_handler,
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", "8000"))),
    )
