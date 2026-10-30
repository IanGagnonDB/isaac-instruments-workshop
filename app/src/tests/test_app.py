import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

import app as app_module


@pytest.fixture
def fake_client(monkeypatch):
    client = SimpleNamespace()
    monkeypatch.setattr(app_module, "get_workspace_client", lambda: client)
    monkeypatch.setattr(
        app_module,
        "configured_endpoint_name",
        lambda: "sam3-video-frames-v2",
    )
    monkeypatch.setattr(
        app_module,
        "configured_transcript_endpoint_name",
        Mock(
            side_effect=lambda model: {
                "luna": "databricks-gpt-5-6-luna",
                "gemini": "databricks-gemini-3-8-flash",
            }[model]
        ),
    )
    monkeypatch.setattr(app_module, "metadata_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(app_module, "ensure_directory", AsyncMock())
    monkeypatch.setattr(app_module, "download_to_local_file", Mock(return_value=100))
    monkeypatch.setattr(app_module, "upload_local_file", Mock())
    monkeypatch.setattr(app_module, "upload_text", Mock())
    monkeypatch.setattr(
        app_module,
        "create_visual_transcript",
        AsyncMock(return_value="**Vehicle detected** moving through the scene."),
    )
    monkeypatch.setattr(
        app_module,
        "segment_video",
        Mock(
            return_value=SimpleNamespace(
                sampled_frame_count=12,
                matched_frame_count=4,
                transcript_frames=(),
            )
        ),
    )
    return client


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as value:
        yield value


def test_safe_video_name_rejects_paths_and_unknown_types():
    with pytest.raises(HTTPException):
        app_module.safe_video_name("../private.mp4")
    with pytest.raises(HTTPException):
        app_module.safe_video_name("notes.txt")

    assert app_module.rooted_path(app_module.INPUTS_DIR, "camera-01.MOV").endswith(
        "/inputs/camera-01.MOV"
    )


def test_browser_upload_name_uses_utc_minute_prefix(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "now_utc",
        lambda: datetime(2026, 9, 16, 16, 0, tzinfo=timezone.utc),
    )

    assert (
        app_module.upload_video_name("Original-filename.MP4")
        == "2026-09-16T1600-Original-filename.MP4"
    )


def test_run_submission_has_one_set_of_defaults():
    submission = app_module.RunSubmission.parse(
        {"source_name": "camera.mp4", "prompt": "white truck"}
    )

    assert submission.frame_stride == 15
    assert submission.truncate is True
    assert submission.threshold == 0.5
    assert submission.output_fps == 8
    assert submission.transcript_model == "luna"


def test_configured_endpoint_names_use_all_isolated_shards(monkeypatch):
    for index in range(1, 5):
        monkeypatch.setenv(f"SAM3_ENDPOINT_{index}", f"sam-{index}")

    assert app_module.configured_endpoint_names() == (
        "sam-1",
        "sam-2",
        "sam-3",
        "sam-4",
    )


def test_run_submission_accepts_source_timing_and_rejects_unknown_models():
    submission = app_module.RunSubmission.parse(
        {
            "source_name": "camera.mp4",
            "prompt": "white truck",
            "output_fps": None,
            "transcript_model": "gemini",
        }
    )

    assert submission.output_fps is None
    assert submission.transcript_model == "gemini"

    with pytest.raises(HTTPException, match="Luna or Gemini"):
        app_module.RunSubmission.parse(
            {
                "source_name": "camera.mp4",
                "prompt": "white truck",
                "transcript_model": "unknown",
            }
        )


@pytest.mark.anyio
async def test_create_run_segments_live_and_returns_the_output(client, fake_client):
    response = await client.post(
        "/api/runs",
        json={
            "source_name": "traffic.mov",
            "prompt": "person wearing red",
            "frame_stride": 15,
            "truncate": False,
            "threshold": 0.4,
            "output_fps": 12,
            "transcript_model": "gemini",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "output_name": "traffic.mov",
        "media_url": "/api/media/traffic.mov",
        "description": "**Vehicle detected** moving through the scene.",
        "description_error": None,
        "elapsed_seconds": payload["elapsed_seconds"],
        "sampled_frame_count": 12,
        "matched_frame_count": 4,
    }
    assert payload["elapsed_seconds"] >= 1
    args = app_module.segment_video.call_args.args
    assert args[1] == "sam3-video-frames-v2"
    assert app_module.segment_video.call_args.kwargs == {
        "prompt": "person wearing red",
        "frame_stride": 15,
        "threshold": 0.4,
        "truncate": False,
        "output_fps": 12,
        "preview_callback": None,
    }
    app_module.configured_transcript_endpoint_name.assert_called_once_with("gemini")
    app_module.create_visual_transcript.assert_awaited_once()
    assert (
        app_module.create_visual_transcript.await_args.args[1]
        == "databricks-gemini-3-8-flash"
    )
    app_module.upload_text.assert_called_once()


def test_stream_run_sends_previews_then_the_final_output(monkeypatch):
    async def fake_segmentation(submission, preview_callback):
        preview_callback(
            {
                "type": "preview",
                "preview_b64": "cHJldmlldw==",
                "processed_frame_count": 1,
                "sampled_frame_count": 2,
                "matched_frame_count": 1,
            }
        )
        return {
            "output_name": submission.source_name,
            "media_url": "/api/media/traffic.mov",
            "description": None,
            "elapsed_seconds": 4,
        }

    monkeypatch.setattr(app_module, "perform_live_segmentation", fake_segmentation)
    with TestClient(app_module.app) as test_client:
        with test_client.websocket_connect("/api/segment-stream") as socket:
            socket.send_json(
                {"source_name": "traffic.mov", "prompt": "person wearing red"}
            )
            assert socket.receive_json()["type"] == "preview"
            complete = socket.receive_json()

    assert complete["type"] == "complete"
    assert complete["output"]["media_url"] == "/api/media/traffic.mov"


@pytest.mark.anyio
async def test_invalid_run_request_returns_stable_error_shape(client):
    response = await client.post(
        "/api/runs",
        json={"source_name": "../traffic.mov", "prompt": ""},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {"message": "A video filename is required."}
    }


@pytest.mark.anyio
async def test_input_listing_identifies_its_configured_volume_root(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "list_video_files",
        AsyncMock(
            return_value=[
                {"name": "camera.mov", "size_bytes": 10_485_760},
                {"name": "traffic.mp4", "size_bytes": 25_165_824},
            ]
        ),
    )

    response = await client.get("/api/inputs?q=cam")

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {"name": "camera.mov", "size_bytes": 10_485_760},
            {"name": "traffic.mp4", "size_bytes": 25_165_824},
        ],
        "root": app_module.INPUTS_DIR,
    }


@pytest.mark.anyio
async def test_video_listing_includes_file_size(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "call_sdk",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    name="small.mp4",
                    file_size=1_572_864,
                    is_directory=False,
                ),
                SimpleNamespace(
                    name="notes.txt",
                    file_size=128,
                    is_directory=False,
                ),
            ]
        ),
    )

    assert await app_module.list_video_files(app_module.INPUTS_DIR) == [
        {"name": "small.mp4", "size_bytes": 1_572_864}
    ]


@pytest.mark.anyio
async def test_missing_static_asset_does_not_return_the_spa(client):
    response = await client.get("/favicon.ico")

    assert response.status_code == 404
    assert response.json() == {"error": {"message": "Static asset not found."}}


@pytest.mark.anyio
async def test_http_telemetry_preserves_valid_trace_id(client):
    response = await client.get(
        "/api/health",
        headers={"X-Trace-ID": "browser-run-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Trace-ID"] == "browser-run-123"


@pytest.mark.anyio
async def test_browser_telemetry_is_validated_and_logged(client, caplog):
    with caplog.at_level("INFO", logger="cv-video-app"):
        response = await client.post(
            "/api/telemetry",
            headers={"X-Trace-ID": "browser-run-456"},
            json={
                "event": "workflow_completed",
                "fields": {"duration_ms": 1234, "delivery_mode": "preview"},
            },
        )

    assert response.status_code == 204
    event = next(
        json.loads(record.message)
        for record in caplog.records
        if '"event":"client.workflow_completed"' in record.message
    )
    assert event["trace_id"] == "browser-run-456"
    assert event["client_fields"]["duration_ms"] == 1234



SAMPLE_EVENT = {
    "event_id": "EVT-0001",
    "event_type": "stop_sign_violation",
    "event_label": "Rolling stop at stop sign",
    "clip": "nexar_negative_01041.mp4",
    "vehicle_id": "TRK-0412",
    "recorded_at": "2026-10-21T14:32:08Z",
    "edge_severity": "medium",
}


def test_event_ids_reject_paths():
    assert app_module.safe_event_id("EVT-0001") == "EVT-0001"
    for invalid in ("", "../x", "a/b", "x" * 65, "-leading"):
        with pytest.raises(HTTPException):
            app_module.safe_event_id(invalid)


def test_table_names_are_quoted_and_validated():
    assert app_module.quoted_table("cat.sch.edge_events") == "`cat`.`sch`.`edge_events`"
    for invalid in ("edge_events", "a.b.c; DROP TABLE x", "a.b.`c`"):
        with pytest.raises(HTTPException):
            app_module.quoted_table(invalid)


@pytest.mark.anyio
async def test_event_list_reads_review_status_from_delta(client, monkeypatch):
    run_sql = Mock(
        return_value=[
            {**SAMPLE_EVENT, "review_status": "reviewed", "reviewed_severity": "high",
             "edge_classification_correct": "false"},
        ]
    )
    monkeypatch.setattr(app_module, "run_sql", run_sql)

    response = await client.get("/api/events")

    assert response.status_code == 200
    [item] = response.json()["items"]
    assert item["review_status"] == "reviewed"
    assert item["edge_classification_correct"] is False
    assert "edge_events" in run_sql.call_args.args[0]


@pytest.mark.anyio
async def test_event_detail_returns_payload_reviews_and_event_types(client, fake_client, monkeypatch):
    def fake_sql(statement, parameters=None):
        assert parameters == {"event_id": "EVT-0001"}
        if "payload" in statement:
            return [{"payload": json.dumps(SAMPLE_EVENT), "review_status": "pending"}]
        return []

    monkeypatch.setattr(app_module, "run_sql", fake_sql)

    response = await client.get("/api/events/EVT-0001")

    assert response.status_code == 200
    body = response.json()
    assert body["event"]["clip"] == "nexar_negative_01041.mp4"
    assert body["clip_url"] == "/api/source-media/nexar_negative_01041.mp4"
    assert body["output_url"] == "/api/media/nexar_negative_01041.mp4"
    assert "no_event" in body["event_types"]


@pytest.mark.anyio
async def test_review_inserts_history_and_marks_event_reviewed(client, monkeypatch):
    calls = []

    def fake_sql(statement, parameters=None):
        calls.append((statement, parameters))
        if statement.lstrip().startswith("SELECT"):
            return [{"event_type": "harsh_acceleration", "edge_severity": "low"}]
        return []

    monkeypatch.setattr(app_module, "run_sql", fake_sql)

    response = await client.post(
        "/api/events/EVT-0001/reviews",
        headers={"X-Forwarded-Email": "reviewer@example.com"},
        json={
            "severity": "None",
            "edge_classification_correct": False,
            "corrected_event_type": "no_event",
            "notes": "Normal merge.",
            "sam3_prompt": "vehicle",
            "sam3_matched_frames": 40,
            "sam3_sampled_frames": 41,
        },
    )

    assert response.status_code == 201
    assert response.json()["reviewer"] == "reviewer@example.com"
    insert, update = calls[1], calls[2]
    assert "INSERT INTO" in insert[0] and "event_reviews" in insert[0]
    assert insert[1]["edge_event_type"] == "harsh_acceleration"
    assert insert[1]["corrected"] == "no_event"
    assert insert[1]["edge_correct"] is False
    assert "UPDATE" in update[0] and "review_status = 'reviewed'" in update[0]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"severity": "extreme", "edge_classification_correct": True}, "Severity must be one of"),
        ({"severity": "low", "edge_classification_correct": False, "notes": "x"},
         "Choose what the event actually was"),
        ({"severity": "low", "edge_classification_correct": False,
          "corrected_event_type": "no_event"}, "Add a note"),
    ],
)
async def test_review_validation(client, monkeypatch, body, message):
    run_sql = Mock()
    monkeypatch.setattr(app_module, "run_sql", run_sql)

    response = await client.post("/api/events/EVT-0001/reviews", json=body)

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
    run_sql.assert_not_called()


@pytest.mark.anyio
async def test_index_html_is_not_cached(client, tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)

    response = await client.get("/")

    assert response.headers["cache-control"] == "no-cache"
