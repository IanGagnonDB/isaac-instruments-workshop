import './engineers-corner-slide.css'

const engineeringControls = [
  {
    title: 'Frame stride vs detection recall',
    detail: 'Fewer frames reduce cost; dense sampling catches brief events.',
  },
  {
    title: 'Payload size vs quality',
    detail: 'JPEG quality and resize settings must preserve useful detail.',
  },
  {
    title: 'Batching vs rate limits',
    detail: 'Bound concurrency to protect serving endpoints and latency.',
  },
  {
    title: 'Long-running preview channel',
    detail: 'Use WebSockets so large videos can continue beyond the HTTP timeout.',
  },
  {
    title: 'Idempotency and retries',
    detail: 'Key work by run and frame; retry only transient failures.',
  },
  {
    title: 'UC Volume governance',
    detail: 'Keep inputs and outputs access-controlled and traceable.',
  },
  {
    title: 'Human verification',
    detail: 'Route low-confidence or safety-critical findings to review.',
  },
]

function StageIcon({
  children,
  variant,
}: {
  children: string
  variant: string
}) {
  return (
    <span
      aria-hidden="true"
      className={`engineers-corner__icon engineers-corner__icon--${variant}`}
    >
      {children}
    </span>
  )
}

export function EngineersCornerSlide() {
  return (
    <section
      aria-labelledby="engineers-corner-title"
      className="engineers-corner"
    >
      <header className="engineers-corner__header">
        <div>
          <p className="engineers-corner__eyebrow">Engineer&apos;s Corner</p>
          <h2 id="engineers-corner-title">
            From video bytes to governed insight
          </h2>
        </div>
        <p className="engineers-corner__intro">
          A reference processing pattern—not a claim about one fixed
          implementation. Every boundary is a choice across fidelity, latency,
          cost, and operational safety.
        </p>
      </header>

      <div
        className="engineers-corner__pipeline-scroll"
        role="region"
        aria-label="Video analysis processing pipeline"
        tabIndex={0}
      >
        <ol className="engineers-corner__pipeline">
          <li className="engineers-corner__stage">
            <span className="engineers-corner__step">01</span>
            <StageIcon variant="source">▶</StageIcon>
            <div>
              <h3>Source ingest</h3>
              <p>Register the video and immutable run identity.</p>
              <span className="engineers-corner__meta">MP4 · MOV · AVI</span>
            </div>
          </li>

          <li className="engineers-corner__stage">
            <span className="engineers-corner__step">02</span>
            <StageIcon variant="frames">▥</StageIcon>
            <div>
              <h3>Sample + buffer</h3>
              <p>Select every Nth frame and retain sequence metadata.</p>
              <span className="engineers-corner__meta">
                frame_index · timestamp
              </span>
            </div>
          </li>

          <li className="engineers-corner__stage engineers-corner__stage--loop">
            <span className="engineers-corner__loop-label">
              Per-frame processing loop
            </span>
            <span className="engineers-corner__step">03</span>
            <StageIcon variant="loop">↻</StageIcon>
            <div>
              <h3>Prepare each frame</h3>
              <ol className="engineers-corner__loop-steps">
                <li>
                  <span>1</span>Convert to JPEG
                </li>
                <li>
                  <span>2</span>Compress / resize
                </li>
                <li>
                  <span>3</span>Base64 encode
                </li>
                <li>
                  <span>4</span>Construct payload
                </li>
              </ol>
            </div>
          </li>

          <li className="engineers-corner__stage engineers-corner__stage--serving">
            <span className="engineers-corner__step">04</span>
            <StageIcon variant="model">◎</StageIcon>
            <div>
              <h3>Segment + serve</h3>
              <p>
                Call segmentation and model-serving endpoints with bounded
                concurrency.
              </p>
              <span className="engineers-corner__meta">retry · dedupe</span>
            </div>
          </li>

          <li className="engineers-corner__stage">
            <span className="engineers-corner__step">05</span>
            <StageIcon variant="assembly">≡</StageIcon>
            <div>
              <h3>Assemble in order</h3>
              <p>Rejoin responses by frame index into temporal context.</p>
              <span className="engineers-corner__meta">stable ordering</span>
            </div>
          </li>

          <li className="engineers-corner__stage engineers-corner__stage--summary">
            <span className="engineers-corner__step">06</span>
            <StageIcon variant="summary">✦</StageIcon>
            <div>
              <h3>Summarize</h3>
              <p>
                Ask a foundation model to explain evidence across the sequence.
              </p>
              <span className="engineers-corner__meta">grounded context</span>
            </div>
          </li>

          <li className="engineers-corner__stage engineers-corner__stage--output">
            <span className="engineers-corner__step">07</span>
            <StageIcon variant="output">✓</StageIcon>
            <div>
              <h3>Govern + observe</h3>
              <p>Publish results with lineage, metrics, traces, and review.</p>
              <span className="engineers-corner__meta">report · audit trail</span>
            </div>
          </li>
        </ol>
      </div>

      <section
        className="engineers-corner__controls"
        aria-labelledby="engineers-corner-controls-title"
      >
        <div className="engineers-corner__controls-heading">
          <span>Design review</span>
          <h3 id="engineers-corner-controls-title">Decisions that matter</h3>
        </div>
        <dl>
          {engineeringControls.map((control) => (
            <div key={control.title}>
              <dt>{control.title}</dt>
              <dd>{control.detail}</dd>
            </div>
          ))}
        </dl>
      </section>
    </section>
  )
}
