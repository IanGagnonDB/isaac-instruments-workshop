export interface OutputDetail {
  output_name: string
  media_url: string
  description: string | null
  description_error?: string | null
  elapsed_seconds?: number
  sampled_frame_count?: number
  matched_frame_count?: number
}

export interface PreviewEvent {
  preview_b64: string
  processed_frame_count: number
  sampled_frame_count: number
  matched_frame_count: number
  current_detection_count: number
  elapsed_seconds: number
  estimated_remaining_seconds: number
}

export interface StatusEvent {
  stage: string
  message: string
}

export interface RunInput {
  source_name: string
  prompt: string
  frame_stride: number
  truncate: boolean
  threshold: number
  output_fps: number | null
  transcript_model: 'luna' | 'gemini'
}

export interface VideoFile {
  name: string
  size_bytes: number | null
}

export interface FileListing {
  items: VideoFile[]
  root: string
}

interface ApiError {
  error?: {
    message?: string
  }
}

type TelemetryFields = Record<string, string | number | boolean | null>

export function createTraceId(): string {
  return typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function sendTelemetry(
  event: string,
  fields: TelemetryFields,
  traceId = createTraceId(),
): void {
  void fetch('/api/telemetry', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Trace-ID': traceId,
    },
    body: JSON.stringify({ event, fields }),
    keepalive: true,
  }).catch(() => {
    // Telemetry must never interrupt the user workflow.
  })
}

async function apiFetch<T>(
  input: string,
  init?: RequestInit,
  traceId = createTraceId(),
): Promise<T> {
  const started = performance.now()
  const headers = new Headers(init?.headers)
  headers.set('X-Trace-ID', traceId)
  const method = init?.method ?? 'GET'
  const path = new URL(input, window.location.origin).pathname
  let response: Response
  try {
    response = await fetch(input, { ...init, headers })
  } catch (error) {
    sendTelemetry(
      'api_request_failed',
      {
        method,
        path,
        duration_ms: Math.round(performance.now() - started),
        error_type: error instanceof Error ? error.name : 'UnknownError',
      },
      traceId,
    )
    throw error
  }
  sendTelemetry(
    'api_request_completed',
    {
      method,
      path,
      status_code: response.status,
      duration_ms: Math.round(performance.now() - started),
    },
    traceId,
  )
  if (!response.ok) {
    let payload: ApiError = {}
    try {
      payload = (await response.json()) as ApiError
    } catch {
      // The proxy can return a non-JSON error page.
    }
    throw new Error(payload.error?.message || `Request failed (${response.status}).`)
  }
  return (await response.json()) as T
}

async function listFiles(
  kind: 'inputs' | 'outputs',
  query = '',
): Promise<FileListing> {
  return apiFetch<FileListing>(
    `/api/${kind}?q=${encodeURIComponent(query)}`,
  )
}

export function listInputs(query = ''): Promise<FileListing> {
  return listFiles('inputs', query)
}

export function listOutputs(query = ''): Promise<FileListing> {
  return listFiles('outputs', query)
}

export function uploadInput(
  file: File,
  onProgress: (percent: number) => void,
): Promise<{ source_name: string; size_bytes: number }> {
  return new Promise((resolve, reject) => {
    const traceId = createTraceId()
    const started = performance.now()
    const request = new XMLHttpRequest()
    request.open('PUT', `/api/inputs/${encodeURIComponent(file.name)}`)
    request.setRequestHeader('Content-Type', file.type || 'application/octet-stream')
    request.setRequestHeader('X-Trace-ID', traceId)
    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    })
    request.addEventListener('load', () => {
      let payload: {
        source_name?: string
        size_bytes?: number
        error?: { message?: string }
      } = {}
      try {
        payload = JSON.parse(request.responseText) as typeof payload
      } catch {
        // Preserve the status-based fallback below.
      }
      if (request.status >= 200 && request.status < 300) {
        sendTelemetry(
          'upload_completed',
          {
            status_code: request.status,
            bytes: file.size,
            duration_ms: Math.round(performance.now() - started),
          },
          traceId,
        )
        resolve({
          source_name: payload.source_name ?? file.name,
          size_bytes: payload.size_bytes ?? file.size,
        })
      } else {
        sendTelemetry(
          'upload_failed',
          {
            status_code: request.status,
            bytes: file.size,
            duration_ms: Math.round(performance.now() - started),
          },
          traceId,
        )
        reject(
          new Error(payload.error?.message || `Upload failed (${request.status}).`),
        )
      }
    })
    request.addEventListener('error', () => {
      sendTelemetry(
        'upload_failed',
        {
          status_code: 0,
          bytes: file.size,
          duration_ms: Math.round(performance.now() - started),
        },
        traceId,
      )
      reject(new Error('The upload connection was interrupted.'))
    })
    request.send(file)
  })
}

export function createRun(input: RunInput, traceId = createTraceId()): Promise<{
  output_name: string
  media_url: string
  description: string | null
  elapsed_seconds: number
  sampled_frame_count: number
  matched_frame_count: number
}> {
  return apiFetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }, traceId)
}

export function streamRun(
  input: RunInput,
  onPreview: (event: PreviewEvent) => void,
  onStatus: (event: StatusEvent) => void,
  traceId = createTraceId(),
): Promise<OutputDetail> {
  return new Promise((resolve, reject) => {
    const started = performance.now()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(
      `${protocol}//${window.location.host}/api/segment-stream?trace_id=${encodeURIComponent(traceId)}`,
    )
    let settled = false

    socket.addEventListener('open', () => {
      sendTelemetry('websocket_opened', {}, traceId)
      socket.send(JSON.stringify(input))
    })
    socket.addEventListener('message', (message) => {
      const event = JSON.parse(message.data as string) as
        | ({ type: 'preview' } & PreviewEvent)
        | ({ type: 'status' } & StatusEvent)
        | { type: 'complete'; output: OutputDetail }
        | { type: 'error'; message: string }
      if (event.type === 'preview') {
        onPreview(event)
      } else if (event.type === 'status') {
        onStatus(event)
      } else if (event.type === 'complete') {
        settled = true
        sendTelemetry(
          'websocket_completed',
          { duration_ms: Math.round(performance.now() - started) },
          traceId,
        )
        resolve(event.output)
        socket.close()
      } else {
        settled = true
        sendTelemetry(
          'websocket_failed',
          {
            duration_ms: Math.round(performance.now() - started),
            reason: 'server_error',
          },
          traceId,
        )
        reject(new Error(event.message))
        socket.close()
      }
    })
    socket.addEventListener('error', () => {
      if (!settled) {
        settled = true
        sendTelemetry(
          'websocket_failed',
          {
            duration_ms: Math.round(performance.now() - started),
            reason: 'connection_error',
          },
          traceId,
        )
        reject(new Error('The live preview connection failed.'))
      }
    })
    socket.addEventListener('close', () => {
      if (!settled) {
        settled = true
        sendTelemetry(
          'websocket_failed',
          {
            duration_ms: Math.round(performance.now() - started),
            reason: 'closed_early',
          },
          traceId,
        )
        reject(new Error('The live preview connection closed before completion.'))
      }
    })
  })
}

export function getOutput(filename: string): Promise<OutputDetail> {
  return apiFetch(`/api/outputs/${encodeURIComponent(filename)}`)
}

export type Severity = 'none' | 'low' | 'medium' | 'high' | 'critical'

export interface EdgeDetection {
  label: string
  confidence: number
  first_seen_s: number
  last_seen_s: number
}

export interface EdgeEvent {
  event_id: string
  simulated?: boolean
  source_note?: string
  clip: string
  device_id?: string
  vehicle_id?: string
  driver_id?: string
  recorded_at?: string
  location?: { lat: number; lon: number } | null
  event_type: string
  event_label?: string
  trigger?: { time_s: number; clip_duration_s?: number }
  edge_detections?: EdgeDetection[]
  edge_severity?: Severity
  telemetry?: {
    speed_kmh?: { t_s: number; speed_kmh: number }[]
    [key: string]: unknown
  }
  suggested_prompts?: string[]
  nexar_label?: {
    collision_or_near_miss: boolean | null
    time_of_event_s: number | null
    scene?: string | null
    weather?: string | null
    light_conditions?: string | null
  }
}

export interface EventSummary {
  event_id: string
  event_type: string
  event_label?: string
  clip: string
  vehicle_id?: string
  recorded_at?: string
  edge_severity?: Severity
  review_status: 'pending' | 'reviewed'
  reviewed_severity: Severity | null
  edge_classification_correct?: boolean | null
  reviewed_by?: string | null
}

export interface EventReview {
  event_id: string
  severity: Severity
  edge_classification_correct: boolean
  corrected_event_type: string | null
  notes: string
  sam3_prompt: string | null
  sam3_matched_frames: number | null
  sam3_sampled_frames: number | null
  reviewer: string
  submitted_at: string
}

export interface EventDetail {
  event: EdgeEvent
  review_status: 'pending' | 'reviewed'
  clip_url: string
  output_url: string | null
  reviews: EventReview[]
  event_types: string[]
}

export function listEvents(): Promise<{ items: EventSummary[]; table: string }> {
  return apiFetch('/api/events')
}

export function getEvent(eventId: string): Promise<EventDetail> {
  return apiFetch(`/api/events/${encodeURIComponent(eventId)}`)
}

export function submitReview(
  eventId: string,
  review: {
    severity: Severity
    edge_classification_correct: boolean
    corrected_event_type?: string | null
    notes: string
    sam3_prompt?: string | null
    sam3_matched_frames?: number | null
    sam3_sampled_frames?: number | null
  },
): Promise<EventReview> {
  return apiFetch(`/api/events/${encodeURIComponent(eventId)}/reviews`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(review),
  })
}
