import { type FormEvent, useEffect, useRef, useState } from 'react'
import {
  type EdgeEvent,
  type EventDetail,
  type EventSummary,
  type Severity,
  getEvent,
  listEvents,
  submitReview,
} from './api'
import './review.css'

const SEVERITIES: { value: Severity; label: string; hint: string }[] = [
  { value: 'none', label: 'None', hint: 'Not a safety event' },
  { value: 'low', label: 'Low', hint: 'Coach later' },
  { value: 'medium', label: 'Medium', hint: 'Coach this week' },
  { value: 'high', label: 'High', hint: 'Coach today' },
  { value: 'critical', label: 'Critical', hint: 'Escalate now' },
]

function humanize(value: string | undefined | null): string {
  if (!value) return '—'
  return value.replace(/_/g, ' ').replace(/^\w/, (letter) => letter.toUpperCase())
}

function formatTimestamp(value: string | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function SpeedTrace({ event }: { event: EdgeEvent }) {
  const points = event.telemetry?.speed_kmh ?? []
  const [hover, setHover] = useState<number | null>(null)
  if (points.length < 2) return null

  const width = 360
  const height = 96
  const pad = { top: 16, right: 10, bottom: 20, left: 30 }
  const maxT = points[points.length - 1].t_s
  const maxV = Math.max(10, Math.ceil(Math.max(...points.map((p) => p.speed_kmh)) / 10) * 10)
  const x = (t: number) => pad.left + (t / maxT) * (width - pad.left - pad.right)
  const y = (v: number) => pad.top + (1 - v / maxV) * (height - pad.top - pad.bottom)
  const path = points
    .map((p, index) => `${index ? 'L' : 'M'}${x(p.t_s).toFixed(1)},${y(p.speed_kmh).toFixed(1)}`)
    .join(' ')
  const trigger = event.trigger?.time_s
  const minSpeed = Math.min(...points.map((p) => p.speed_kmh))
  const hovered = hover === null ? null : points[hover]

  return (
    <figure className="speed-trace">
      <figcaption>
        <strong>Vehicle speed (km/h)</strong>
        <span>
          Min {minSpeed} · Max {Math.max(...points.map((p) => p.speed_kmh))}
          {trigger !== undefined && ` · Trigger at ${trigger}s`}
        </span>
      </figcaption>
      <svg
        aria-label={`Speed over ${maxT} seconds, minimum ${minSpeed} kilometres per hour`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(mouse) => {
          const box = mouse.currentTarget.getBoundingClientRect()
          const t = ((mouse.clientX - box.left) / box.width) * width
          const seconds = ((t - pad.left) / (width - pad.left - pad.right)) * maxT
          const index = Math.round(Math.min(Math.max(seconds, 0), maxT))
          setHover(points.findIndex((p) => p.t_s === index))
        }}
        role="img"
        viewBox={`0 0 ${width} ${height}`}
      >
        {[0, maxV / 2, maxV].map((v) => (
          <g key={v}>
            <line className="speed-grid" x1={pad.left} x2={width - pad.right} y1={y(v)} y2={y(v)} />
            <text className="speed-axis" x={pad.left - 6} y={y(v) + 3} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {[0, maxT / 2, maxT].map((t) => (
          <text className="speed-axis" key={t} x={x(t)} y={height - 5} textAnchor="middle">
            {t}s
          </text>
        ))}
        {trigger !== undefined && (
          <g>
            <line className="speed-trigger" x1={x(trigger)} x2={x(trigger)} y1={pad.top} y2={height - pad.bottom} />
            <text className="speed-trigger-label" x={x(trigger) + 4} y={pad.top - 5}>
              trigger
            </text>
          </g>
        )}
        <path className="speed-line" d={path} />
        {hovered && (
          <g>
            <line className="speed-crosshair" x1={x(hovered.t_s)} x2={x(hovered.t_s)} y1={pad.top} y2={height - pad.bottom} />
            <circle className="speed-dot" cx={x(hovered.t_s)} cy={y(hovered.speed_kmh)} r={4} />
          </g>
        )}
      </svg>
      <p className="speed-readout" aria-live="polite">
        {hovered ? `${hovered.t_s}s: ${hovered.speed_kmh} km/h` : 'Hover the trace for values'}
      </p>
    </figure>
  )
}

export function EventReviewPage({
  onAnalyze,
}: {
  onAnalyze: (clip: string, prompt: string) => void
}) {
  const [events, setEvents] = useState<EventSummary[]>([])
  const [listError, setListError] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [detail, setDetail] = useState<EventDetail | null>(null)
  const [detailError, setDetailError] = useState('')
  const [showSegmented, setShowSegmented] = useState(false)
  const [severity, setSeverity] = useState<Severity | null>(null)
  const [edgeCorrect, setEdgeCorrect] = useState<boolean | null>(null)
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitMessage, setSubmitMessage] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)

  function refreshList() {
    listEvents()
      .then((listing) => {
        setEvents(listing.items)
        setListError('')
        setSelectedId((current) => current || listing.items[0]?.event_id || '')
      })
      .catch((error: Error) => setListError(error.message))
  }

  useEffect(refreshList, [])

  useEffect(() => {
    if (!selectedId) return
    let active = true
    setDetail(null)
    setDetailError('')
    setSeverity(null)
    setEdgeCorrect(null)
    setNotes('')
    setSubmitMessage('')
    setShowSegmented(false)
    getEvent(selectedId)
      .then((value) => active && setDetail(value))
      .catch((error: Error) => active && setDetailError(error.message))
    return () => {
      active = false
    }
  }, [selectedId])

  async function handleSubmit(form: FormEvent) {
    form.preventDefault()
    if (!detail || !severity || edgeCorrect === null) return
    setSubmitting(true)
    setSubmitMessage('')
    try {
      const review = await submitReview(detail.event.event_id, {
        severity,
        edge_classification_correct: edgeCorrect,
        notes,
        sam3_prompt: detail.event.suggested_prompts?.[0],
      })
      setDetail({ ...detail, reviews: [...detail.reviews, review] })
      setSubmitMessage(`Grade submitted by ${review.reviewer}.`)
      refreshList()
    } catch (error) {
      setSubmitMessage(error instanceof Error ? error.message : 'The grade could not be saved.')
    } finally {
      setSubmitting(false)
    }
  }

  function jumpToTrigger() {
    const trigger = detail?.event.trigger?.time_s
    if (videoRef.current && trigger !== undefined) {
      videoRef.current.currentTime = Math.max(0, trigger - 3)
      void videoRef.current.play()
    }
  }

  const event = detail?.event
  const latest = detail?.reviews[detail.reviews.length - 1]
  const videoUrl = showSegmented && detail?.output_url ? detail.output_url : detail?.clip_url

  return (
    <div className="review-workspace">
      <section className="panel review-queue" aria-labelledby="queue-heading">
        <div className="panel-heading">
          <div>
            <p className="section-label">Edge events</p>
            <h2 id="queue-heading">Review queue</h2>
          </div>
          <span className="badge badge-running">
            {events.filter((item) => item.review_status === 'pending').length} pending
          </span>
        </div>
        {listError && <p className="field-error review-pad">{listError}</p>}
        {!listError && !events.length && (
          <p className="review-pad review-muted">
            No events yet. Run the setup job to generate the sample events.
          </p>
        )}
        <ul className="queue-list">
          {events.map((item) => (
            <li key={item.event_id}>
              <button
                aria-current={item.event_id === selectedId ? 'true' : undefined}
                className="queue-item"
                onClick={() => setSelectedId(item.event_id)}
                type="button"
              >
                <span className="queue-top">
                  <strong>{item.event_label || humanize(item.event_type)}</strong>
                  <span className={`queue-status queue-status-${item.review_status}`}>
                    {item.review_status === 'reviewed'
                      ? `Graded ${humanize(item.reviewed_severity)}`
                      : 'Pending'}
                  </span>
                </span>
                <span className="queue-meta">
                  {item.event_id} · {item.vehicle_id ?? '—'} · Edge {humanize(item.edge_severity)}
                </span>
                <span className="queue-meta">{formatTimestamp(item.recorded_at)}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel review-detail" aria-labelledby="detail-heading">
        <div className="panel-heading">
          <div>
            <p className="section-label">{event ? event.event_id : 'Event'}</p>
            <h2 id="detail-heading">
              {event ? event.event_label || humanize(event.event_type) : 'Select an event'}
            </h2>
          </div>
          {event?.simulated && <span className="badge review-sim-badge">Simulated metadata</span>}
        </div>

        {detailError && <p className="field-error review-pad">{detailError}</p>}

        {event && detail && (
          <div className="review-body">
            <div className="review-media">
              <div className="video-frame">
                <video controls key={videoUrl} preload="metadata" ref={videoRef}>
                  <source src={videoUrl} />
                </video>
                <span className="video-label">
                  {showSegmented && detail.output_url ? 'SAM3 segmented clip' : 'Edge clip'}
                </span>
              </div>
              <div className="review-actions">
                <button className="button-secondary" onClick={jumpToTrigger} type="button">
                  Jump to trigger
                </button>
                <button
                  className="button-secondary"
                  disabled={!detail.output_url}
                  onClick={() => setShowSegmented((value) => !value)}
                  title={detail.output_url ? '' : 'Run SAM3 on this clip first'}
                  type="button"
                >
                  {showSegmented ? 'Show edge clip' : 'Show SAM3 result'}
                </button>
                <button
                  className="button-primary"
                  onClick={() => onAnalyze(event.clip, event.suggested_prompts?.[0] ?? 'vehicle')}
                  type="button"
                >
                  Analyze with SAM3
                </button>
              </div>
              {event.suggested_prompts?.length ? (
                <p className="review-muted review-prompts">
                  Suggested prompts:{' '}
                  {event.suggested_prompts.map((prompt) => (
                    <button
                      className="prompt-chip"
                      key={prompt}
                      onClick={() => onAnalyze(event.clip, prompt)}
                      type="button"
                    >
                      {prompt}
                    </button>
                  ))}
                </p>
              ) : null}
            </div>

            <div className="review-facts">
              <dl className="fact-grid">
                <div>
                  <dt>Event type</dt>
                  <dd>{humanize(event.event_type)}</dd>
                </div>
                <div>
                  <dt>Edge severity</dt>
                  <dd>{humanize(event.edge_severity)}</dd>
                </div>
                <div>
                  <dt>Vehicle / driver</dt>
                  <dd>
                    {event.vehicle_id ?? '—'} / {event.driver_id ?? '—'}
                  </dd>
                </div>
                <div>
                  <dt>Recorded</dt>
                  <dd>{formatTimestamp(event.recorded_at)}</dd>
                </div>
                <div>
                  <dt>Device</dt>
                  <dd>{event.device_id ?? '—'}</dd>
                </div>
                <div>
                  <dt>Location</dt>
                  <dd>
                    {event.location
                      ? `${event.location.lat.toFixed(4)}, ${event.location.lon.toFixed(4)}`
                      : '—'}
                  </dd>
                </div>
              </dl>

              <SpeedTrace event={event} />

              <table className="detections-table">
                <caption>Edge detections</caption>
                <thead>
                  <tr>
                    <th scope="col">Object</th>
                    <th scope="col">Confidence</th>
                    <th scope="col">Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {(event.edge_detections ?? []).map((item) => (
                    <tr key={item.label}>
                      <td>{humanize(item.label)}</td>
                      <td>{Math.round(item.confidence * 100)}%</td>
                      <td>
                        {item.first_seen_s}s to {item.last_seen_s}s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {event.nexar_label && (
                <p className="review-muted">
                  Dataset label (Nexar):{' '}
                  {event.nexar_label.collision_or_near_miss
                    ? `collision or near-miss at ${event.nexar_label.time_of_event_s}s`
                    : 'normal driving'}
                  {event.nexar_label.scene && ` · ${event.nexar_label.scene}`}
                  {event.nexar_label.weather && ` · ${event.nexar_label.weather}`}
                </p>
              )}
            </div>

            <form className="grade-form" onSubmit={handleSubmit}>
              <fieldset>
                <legend>Severity grade</legend>
                <div className="severity-options">
                  {SEVERITIES.map((option) => (
                    <label
                      className={`severity-option ${severity === option.value ? 'selected' : ''}`}
                      key={option.value}
                    >
                      <input
                        checked={severity === option.value}
                        name="severity"
                        onChange={() => setSeverity(option.value)}
                        type="radio"
                        value={option.value}
                      />
                      <strong>{option.label}</strong>
                      <span>{option.hint}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
              <fieldset>
                <legend>Was the edge classification correct?</legend>
                <div className="edge-choice">
                  {([
                    [true, 'Yes'],
                    [false, 'No, misclassified'],
                  ] as const).map(([value, label]) => (
                    <label className={edgeCorrect === value ? 'selected' : ''} key={label}>
                      <input
                        checked={edgeCorrect === value}
                        name="edge-correct"
                        onChange={() => setEdgeCorrect(value)}
                        type="radio"
                      />
                      {label}
                    </label>
                  ))}
                </div>
              </fieldset>
              <label className="notes-field">
                <span>Reviewer notes</span>
                <textarea
                  maxLength={2000}
                  onChange={(change) => setNotes(change.target.value)}
                  placeholder="What happened, and what should the driver hear?"
                  rows={3}
                  value={notes}
                />
              </label>
              <div className="grade-submit">
                <button
                  className="button-primary"
                  disabled={!severity || edgeCorrect === null || submitting}
                  type="submit"
                >
                  {submitting ? 'Submitting…' : 'Submit grade'}
                </button>
                {submitMessage && (
                  <span className="review-muted" role="status">
                    {submitMessage}
                  </span>
                )}
              </div>
              {latest && (
                <p className="review-muted">
                  Last grade: <strong>{humanize(latest.severity)}</strong> by {latest.reviewer},{' '}
                  {formatTimestamp(latest.submitted_at)} ({detail.reviews.length} total)
                </p>
              )}
            </form>
          </div>
        )}
      </section>
    </div>
  )
}
