# SAM3 video segmentation app

A Databricks App that finds anything you can describe in a video. Upload a clip (or pick one from a Unity Catalog Volume), type a short prompt such as `pedestrian` or `open car door`, and get back:

- an MP4 with segmentation masks drawn on every match, with live previews while it runs
- a written narrative of what happened, from a vision-language model

The app opens on a **Review queue** that mirrors a fleet-safety workflow: the edge device flags an event and uploads a short clip plus JSON metadata; a reviewer in the cloud watches the clip, checks the edge detections and speed trace, optionally runs SAM3 on it right there, then submits a severity grade. Grades land in Delta tables and an AI/BI dashboard tracks where the edge misclassifies. A **Free search** tab keeps the open-ended "find anything in any video" demo.

| Part | What it does | Where it runs |
|---|---|---|
| App (`src/`) | UI, frame sampling, overlays, MP4 encoding | Databricks Apps (CPU) |
| SAM3 (`facebook/sam3`) | Finds and outlines objects that match the text prompt. No training needed. | Model Serving GPU endpoint |
| Narrative model | Describes up to 24 matched frames in plain language | Foundation Model APIs (pay-per-token) |
| Volume | `inputs/`, `outputs/`, `descriptions/`, `events/`, `model_checkpoints/` | Unity Catalog |
| Delta tables | `edge_events` (queue and review status), `event_reviews` (grade history) | Unity Catalog, via a serverless SQL warehouse |
| AI/BI dashboard | Edge classification quality | Databricks SQL |

Everything deploys with a Databricks Asset Bundle. No GPU cluster is needed: the only GPU sits behind Model Serving.

---

## 1. Prerequisites

Work through this list before you deploy. Most failures come from a missed step here.

### 1.1 Hugging Face access (do this first, approval can take time)

1. Create or sign in to a [Hugging Face](https://huggingface.co) account.
2. **Request access to SAM3**: open [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), fill in the form and submit. Meta reviews requests. You are ready when the page says **"You have been granted access to this model"**. Review Meta's SAM licence with your legal team before any production use.
3. **Accept the sample-data terms**: open [huggingface.co/datasets/nexar-ai/nexar_collision_prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) and click **Agree and access** (approved instantly). The Nexar Open Data License allows use with attribution. Do not sell, sublicense or redistribute the dataset for profit.
4. **Create a token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Either:
   - token type **Read**, or
   - a fine-grained token with only **"Read access to contents of all public gated repos you can access"** checked.

   Use the same account as steps 2 and 3. Do not give the token write access.

### 1.2 Databricks workspace

- Unity Catalog enabled, with serverless jobs and Databricks Apps available.
- **GPU Model Serving** available in your region. Pick the GPU type for your cloud:

  | Cloud | `gpu_workload_type` | GPU |
  |---|---|---|
  | AWS | `GPU_MEDIUM` | 1 x A10G (recommended) |
  | Azure | `GPU_SMALL` | 1 x T4, 16 GB (works, slower) |
  | Azure | `GPU_LARGE` | 1 x A100, 80 GB (fast, more expensive) |

  Azure has no `GPU_MEDIUM`.
- **Vision models** for the narrative. Check what your workspace has:

  ```bash
  databricks serving-endpoints list -p <profile> | grep -E "gemini|gpt-5"
  ```

  The defaults are `databricks-gemini-3-8-flash` and `databricks-gpt-5-6-luna`. If either is missing, set `fmapi_model` / `luna_model` to models you do have (deploy fails if a named endpoint does not exist).
- **Outbound internet to `huggingface.co`** from serverless compute, for the one-time model and sample download. If your workspace has a serverless network policy, allow this host.
- Permissions for the person deploying: `USE CATALOG` and `USE SCHEMA`, `CREATE MODEL`, `CREATE TABLE` and `CREATE VOLUME` on the schema (or an existing volume with `WRITE VOLUME`), permission to create serving endpoints, apps, jobs, dashboards, secret scopes and a **serverless SQL warehouse** (the bundle creates a 2X-Small one that auto-stops after 10 idle minutes).
- A recent [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html), authenticated to the workspace (tested with v1.17.0):

  ```bash
  databricks auth login --host https://<your-workspace-url> --profile <profile>
  ```

### 1.3 Azure Canada Central notes

- Databricks Apps and GPU Model Serving are supported in Canada Central. Segmentation runs in region.
- Serverless GPU compute (AI Runtime) is not available in Canada. This app does not need it.
- Pay-per-token foundation models (the narrative step) require **cross-geography routing** and the ADI Services integration in your account settings. With that on, the up to 24 narrative frames per video are processed outside Canada. That is a data-residency decision for your organisation. If it is not allowed, the segmentation still works; leave the narrative off.

---

## 2. Store the Hugging Face token in Databricks

Run these in a normal terminal. `put-secret` prompts for the value; paste the token there so it never lands in a file or your shell history.

```bash
databricks secrets create-scope sam3-video-app -p <profile>
databricks secrets put-secret sam3-video-app hf_token -p <profile>
```

Nothing in this repo contains a token. The jobs read it at runtime with `dbutils.secrets.get`. The app itself never uses it.

## 3. Create the schema and volume

Skip if you already have them.

```bash
databricks schemas create <schema> <catalog> -p <profile>
databricks volumes create <catalog> <schema> auto_segment MANAGED -p <profile>
```

## 4. Deploy with Asset Bundles

From this `app/` folder:

```bash
cd app

databricks bundle deploy -p <profile> \
  --var catalog=<catalog> \
  --var schema=<schema> \
  --var gpu_workload_type=GPU_SMALL      # AWS: GPU_MEDIUM
```

To avoid repeating the variables, put them in `.databricks/bundle/dev/variable-overrides.json` (not committed):

```json
{ "catalog": "<catalog>", "schema": "<schema>", "gpu_workload_type": "GPU_SMALL" }
```

This creates the app (not started yet), five jobs, a serverless SQL warehouse and the AI/BI dashboard. The SAM3 endpoint and the Delta tables do not exist yet; the setup job creates them.

| Variable | Default | Notes |
|---|---|---|
| `catalog`, `schema` | required | Where the volume and registered model live |
| `gpu_workload_type` | required | See the table in 1.2 |
| `volume` | `auto_segment` | Must already exist |
| `scale_to_zero` | `"false"` | Endpoint stays warm. See [Cost](#cost-and-scale-to-zero) |
| `sam3_endpoint` | `sam3-video-frames` | Serving endpoint name |
| `app_name` | `sam3-video-segmentation` | 30 characters or fewer |
| `secret_scope`, `secret_key` | `sam3-video-app`, `hf_token` | Where the token lives |
| `fmapi_model`, `luna_model` | Gemini 3.8 Flash, GPT-5.6 Luna | Narrative models |
| `sample_clips_per_class` | `"15"` | Nexar clips per class |

## 5. Run the setup job

```bash
databricks bundle run setup -p <profile>     # plus the same --var flags if you did not use the overrides file
```

It runs three tasks:

1. `register_model`: downloads the SAM3 checkpoint (about 3.4 GB) from Hugging Face into `<volume>/model_checkpoints/`, packages it with MLflow and registers `<catalog>.<schema>.transformers-sam3-frames-v2`. The checkpoint ships inside the model, so the endpoint never calls Hugging Face.
2. `deploy_endpoint`: creates the SAM3 GPU endpoint and grants the app's service principal `CAN_QUERY` on it. Building the serving container is the slow part, usually 20 to 40 minutes.
3. `fetch_sample_clips`: copies 30 Nexar dashcam clips (15 collision or near-miss, 15 normal driving) into `<volume>/inputs/`, and the dataset licence and Nexar's per-clip labels into `<volume>/datasets/nexar/`.
4. `generate_sample_events`: writes seven simulated edge events to `<volume>/events/` (see [Event Review](#event-review)).
5. `load_events`: creates the `edge_events` and `event_reviews` Delta tables, loads the event JSON into `edge_events` as `pending`, and grants the app `SELECT`/`MODIFY` on both.

Plan for 30 to 60 minutes in total. Rerunning is safe: the checkpoint is reused and the endpoint is updated in place.

## 6. Start the app

```bash
databricks bundle run sam3_video_app -p <profile>
```

The command prints the app URL. Share it with workshop users through the app's permissions in the workspace UI.

## 7. Try it

Pick a clip from the inputs list and try short, concrete noun phrases:

| Prompt | Good clip | Why |
|---|---|---|
| `car` | any | Almost every frame matches. Shows the overlays and tracking. |
| `open car door` | `nexar_positive_00004` (near the end) | A door opens into traffic. Nobody trained a model for that. |
| `pedestrian` | `nexar_positive_00007` | Busy crosswalks, vulnerable road users |
| `brake lights` | `nexar_negative_01040` | Night braking, following distance |
| `traffic light`, `crosswalk` | `nexar_negative_01043`, `nexar_positive_00000` | Compliance-style cues |

SAM3 finds things; it does not judge them. Prompts like "dangerous situation" do not work. Leave **truncate** off to see the full clip. One 40-second clip took about 27 seconds end to end on a warm A10G, including the narrative.

## Event Review

The **Review queue** tab (the app's home page) lists edge events from the `edge_events` table, pending first, with their review status.

For each event the reviewer sees the edge clip (with **Jump to trigger**), the edge device's detections and severity estimate, a speed trace with the trigger marked, and Nexar's own label for the clip. **Analyze with SAM3** runs SAM3 on that clip in place, using an editable prompt pre-filled from the event (suggested prompts are one click). Live annotated frames stream in, then the segmented clip and the narrative appear next to the grading form.

The reviewer then picks a severity (None, Low, Medium, High, Critical) and says whether the edge classification was correct. **No, misclassified** asks what the event actually was (from a fixed list of event types) and requires a note. On submit:

- a row is appended to **`event_reviews`** (edge type and severity, reviewer severity, correct or not, corrected type, notes, the SAM3 prompt and frame counts, reviewer email from the Databricks Apps identity headers, timestamp), and
- the event's row in **`edge_events`** is marked `review_status = 'reviewed'` with the latest severity, correction, reviewer and time. Regrading appends another history row and updates the status.

The seven sample events are **simulated** (IDs, speeds, detections and severities are invented for the workshop, and each JSON says so). Their clips and trigger times line up with what happens in the footage:

| Event | Clip | Simulated edge event |
|---|---|---|
| EVT-0101 | `nexar_negative_01041` | Rolling stop at a stop sign |
| EVT-0102 | `nexar_positive_00004` | Parked car door opened into the lane |
| EVT-0103 | `nexar_positive_00007` | Following too close, hard stop |
| EVT-0104 | `nexar_positive_00003` | Harsh braking at night in rain |
| EVT-0105 | `nexar_positive_00008` | Pedestrian crossing at a night intersection |
| EVT-0106 | `nexar_positive_00000` | Van cut in from the left |
| EVT-0107 | `nexar_negative_01043` | "Harsh acceleration" that is really normal driving (an edge false positive to grade as misclassified, actually no event) |

### Dashboard

The bundle deploys an AI/BI dashboard, **Edge event review quality**, on the same warehouse: edge events, reviewed and misclassified counts, misclassification rate, misclassified events by edge event type, an "edge said vs reviewer said" table, every misclassified event with the reviewer's note, and edge vs reviewer severity. It reads `edge_events` and `event_reviews` in the bundle's catalog and schema.

### Bring your own events

Drop a clip into `<volume>/inputs/` and an event JSON into `<volume>/events/` with the same fields (`event_id`, `clip`, `event_type`, `edge_detections`, `telemetry.speed_kmh`, `trigger.time_s`, and so on; see `notebooks/generate_sample_events.py`). The **`sam3-ingest-edge-events`** job has a file-arrival trigger on that folder and loads new events into `edge_events` as `pending` (existing rows and their review status are never overwritten). Development-mode deploys leave triggers paused: unpause the job in the Jobs UI, or run it by hand.

---

## Cost and scale-to-zero

The SAM3 endpoint is **always on by default**, so the app responds immediately. You pay for one GPU for as long as the endpoint is up.

To let it scale to zero when idle (the first request after idling then waits several minutes for a cold start):

```bash
databricks bundle deploy -p <profile> --var scale_to_zero=true
databricks bundle run update_endpoint -p <profile>
```

Switch it back with `scale_to_zero=false` before a live session.

## A second, parallel deployment

To try app changes without touching an existing deployment, use the `preview` target. It deploys a separate app (`sam3-video-review`) and jobs with their own bundle state, and reuses the same volume and SAM3 endpoint:

```bash
databricks bundle deploy -t preview -p <profile> --var ...
databricks bundle run -t preview sam3_video_app -p <profile> --var ...
```

Then grant the new app access to the existing endpoint (or run its `update_endpoint` job, which also updates the endpoint's configuration), and run only the table step of its setup job so the new app gets access to the Delta tables:

```bash
databricks serving-endpoints update-permissions <endpoint-id> -p <profile> \
  --json '{"access_control_list":[{"service_principal_name":"<app service principal client id>","permission_level":"CAN_QUERY"}]}'

databricks jobs run-now -p <profile> --json '{"job_id": <preview setup job id>, "only": ["load_events"]}'
```

## Optional batch jobs

- `sam3-segment-video`: segments one video from `inputs/` as a job (set `trigger_location` and `prompt`), writes the MP4, a manifest with stage timings, and up to 24 frames for narration.
- `sam3-transcribe-video`: writes the narrative for a video that `sam3-segment-video` processed.

## Tear down

```bash
databricks bundle destroy -p <profile>                              # app, jobs, warehouse, dashboard
databricks serving-endpoints delete sam3-video-frames -p <profile>   # created by the setup job, not the bundle
databricks registered-models delete <catalog>.<schema>.transformers-sam3-frames-v2 -p <profile>
databricks tables delete <catalog>.<schema>.edge_events -p <profile>
databricks tables delete <catalog>.<schema>.event_reviews -p <profile>
databricks secrets delete-scope sam3-video-app -p <profile>
```

Then delete the volume contents if you no longer need them. Revoke the Hugging Face token when you are done.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 ... Invalid user token` or `401 ... Please log in` | The token in the secret scope was revoked or regenerated. Store the current token again with `put-secret` (section 2). |
| `GatedRepoError: 403 ... not in the authorized list` | SAM3 access not granted yet, or the dataset terms not accepted, or a fine-grained token without the gated-repos permission. See 1.1. |
| Deploy fails: endpoint `databricks-...` does not exist | Set `fmapi_model` / `luna_model` to models your workspace has. |
| `deploy_endpoint` fails on the workload type | Use `GPU_SMALL` or `GPU_LARGE` on Azure. |
| App says the live SAM3 endpoint is not configured, or requests fail with 403 | Run the setup job (or `update_endpoint`) after deploying, so the app's service principal gets `CAN_QUERY`. |
| First request is very slow | Scale-to-zero is on and the endpoint is cold. |
| Review queue says "No events yet" | Run the setup job (the `generate_sample_events` and `load_events` tasks), or add JSON files to `<volume>/events/` and run `sam3-ingest-edge-events`. |
| Review queue says the review tables could not be queried | The `load_events` task has not run yet, or the app lacks `SELECT`/`MODIFY` on the tables. Rerun `load_events`. |
| A first queue load takes 10 to 20 seconds | The serverless SQL warehouse was stopped and is starting. |
| Download fails or hangs | Serverless egress to `huggingface.co` is blocked. |

## Layout

```
app/
  databricks.yml          bundle: app, warehouse, dashboard, setup, ingest, update_endpoint, batch jobs
  dashboards/             AI/BI dashboard definition
  src/                    app source deployed to Databricks Apps
    app.py                Starlette API and static server
    live_segment.py       SAM3 calls, overlays, MP4 encoding
    visual_transcript.py  narrative generation
    frontend/src/EventReview.tsx  review queue and grading UI
    static/               prebuilt React frontend
    frontend/             React/TypeScript source
    tests/                pytest suite
  notebooks/              setup and batch notebooks
  package.json, vite.config.ts, tsconfig.json   frontend build (optional)
```

To rebuild the frontend after changing `src/frontend/`: `npm ci && npm run build` in `app/` (writes `src/static/`). Frontend tests: `npx vitest run` (on Node 25+, prefix with `NODE_OPTIONS=--no-experimental-webstorage`). Python tests: `pip install -r src/requirements-dev.txt && pytest src/tests`.

## Licence

The app code is provided under the Databricks licence in [LICENSE.md](LICENSE.md) with [NOTICE.md](NOTICE.md). These files were modified from the original Databricks source for this workshop: deployment parameters, single-endpoint default, secret handling, setup job and sample data. SAM3 is subject to Meta's SAM licence; the Nexar clips to the Nexar Open Data License.
