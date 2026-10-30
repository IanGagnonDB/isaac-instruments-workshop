import './use-cases-slide.css'

function PictureValidationIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="use-cases-slide__station-art"
      viewBox="0 0 180 104"
    >
      <rect className="use-cases-slide__art-surface" x="10" y="10" width="98" height="78" rx="3" />
      <path className="use-cases-slide__art-line" d="M22 70 46 46l17 16 13-10 20 18" />
      <circle className="use-cases-slide__art-coral" cx="79" cy="31" r="8" />
      <path className="use-cases-slide__art-focus" d="M17 30V18h12m72 12V18H89M17 68v13h12m72-13v13H89" />
      <rect className="use-cases-slide__art-panel" x="86" y="42" width="82" height="51" rx="3" />
      <circle className="use-cases-slide__art-green" cx="101" cy="58" r="7" />
      <path className="use-cases-slide__art-check" d="m97 58 3 3 6-7" />
      <path className="use-cases-slide__art-rule" d="M115 55h39M115 63h27M98 77h56" />
    </svg>
  )
}

function OccupancyIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="use-cases-slide__station-art"
      viewBox="0 0 180 104"
    >
      <path className="use-cases-slide__art-grid" d="M16 83 71 21l93 31-56 42ZM71 21v52m93-21-93 21m37 21V42" />
      <g className="use-cases-slide__art-people">
        <circle cx="54" cy="50" r="5" />
        <path d="M46 68c1-9 4-13 8-13s7 4 8 13" />
        <circle cx="91" cy="41" r="5" />
        <path d="M83 59c1-9 4-13 8-13s7 4 8 13" />
        <circle cx="125" cy="59" r="5" />
        <path d="M117 77c1-9 4-13 8-13s7 4 8 13" />
      </g>
      <rect className="use-cases-slide__art-panel" x="118" y="10" width="51" height="28" rx="3" />
      <text className="use-cases-slide__art-count" x="129" y="29">03</text>
      <path className="use-cases-slide__art-green-stroke" d="M151 18v13m-6-6.5h12" />
    </svg>
  )
}

function ThreatReviewIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="use-cases-slide__station-art"
      viewBox="0 0 180 104"
    >
      <rect className="use-cases-slide__art-surface" x="10" y="13" width="112" height="76" rx="3" />
      <path className="use-cases-slide__art-muted" d="M10 29h112M20 21h5m5 0h5m5 0h5" />
      <circle className="use-cases-slide__art-person" cx="66" cy="51" r="13" />
      <path className="use-cases-slide__art-person" d="M42 83c3-18 11-27 24-27s21 9 24 27" />
      <path className="use-cases-slide__art-corners" d="M42 47V38h9m39 9V38h-9M42 67v9h9m39-9v9h-9" />
      <path className="use-cases-slide__art-shield" d="M142 12 164 20v18c0 15-9 24-22 30-13-6-22-15-22-30V20Z" />
      <path className="use-cases-slide__art-check" d="m133 38 7 7 13-16" />
      <rect className="use-cases-slide__art-panel" x="104" y="69" width="65" height="24" rx="3" />
      <path className="use-cases-slide__art-rule" d="M114 78h45m-45 7h31" />
    </svg>
  )
}

function ProductionLineIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="use-cases-slide__station-art"
      viewBox="0 0 180 104"
    >
      <path className="use-cases-slide__art-conveyor" d="M11 76h158M22 76v14m136-14v14M18 90h144" />
      <circle className="use-cases-slide__art-wheel" cx="38" cy="83" r="5" />
      <circle className="use-cases-slide__art-wheel" cx="78" cy="83" r="5" />
      <circle className="use-cases-slide__art-wheel" cx="118" cy="83" r="5" />
      <circle className="use-cases-slide__art-wheel" cx="153" cy="83" r="5" />
      <path className="use-cases-slide__art-box" d="m22 52 18-9 18 9v20H22Zm18-9v20m-18-11 18 11 18-11" />
      <rect className="use-cases-slide__art-surface" x="70" y="18" width="38" height="42" rx="3" />
      <path className="use-cases-slide__art-play" d="m85 30 12 9-12 9Z" />
      <path className="use-cases-slide__art-arrow" d="M60 38h8m42 0h8m-4-4 4 4-4 4" />
      <rect className="use-cases-slide__art-panel" x="120" y="18" width="48" height="42" rx="3" />
      <path className="use-cases-slide__art-rule" d="M130 29h28m-28 8h20m-20 8h25" />
      <path className="use-cases-slide__art-green-stroke" d="m143 53 5 5 10-12" />
    </svg>
  )
}

const useCases = [
  {
    number: '01',
    label: 'INPUT / POLICY',
    title: 'Picture policy validation',
    copy:
      'Validate user-provided pictures against approved policy, returning clear feedback for review or correction.',
    className: 'use-cases-slide__station--pictures',
    Illustration: PictureValidationIllustration,
  },
  {
    number: '02',
    label: 'SPACE / AGGREGATE',
    title: 'Privacy-preserving occupancy',
    copy:
      'Segment people in an office to calculate aggregate occupancy—retaining counts, not identity.',
    className: 'use-cases-slide__station--occupancy',
    Illustration: OccupancyIllustration,
  },
  {
    number: '03',
    label: 'STREAM / REVIEW',
    title: 'Authorized threat screening',
    copy:
      'Surface potential matches to known threat actors only in authorized streams, against approved watchlists and always requiring human review.',
    className: 'use-cases-slide__station--threats',
    Illustration: ThreatReviewIllustration,
  },
  {
    number: '04',
    label: 'LINE / VALIDATE',
    title: 'Production-line validation',
    copy:
      'Use a video → text → image workflow to validate manufacturing or USPS-style line stages and exceptions.',
    className: 'use-cases-slide__station--production',
    Illustration: ProductionLineIllustration,
  },
]

function VideoIntelligenceHub() {
  return (
    <div
      aria-label="Central video intelligence workflow"
      className="use-cases-slide__hub"
      role="img"
    >
      <span className="use-cases-slide__hub-ring" aria-hidden="true" />
      <div className="use-cases-slide__hub-screen" aria-hidden="true">
        <span className="use-cases-slide__hub-corner use-cases-slide__hub-corner--one" />
        <span className="use-cases-slide__hub-corner use-cases-slide__hub-corner--two" />
        <span className="use-cases-slide__hub-corner use-cases-slide__hub-corner--three" />
        <span className="use-cases-slide__hub-corner use-cases-slide__hub-corner--four" />
        <span className="use-cases-slide__hub-play">▶</span>
        <span className="use-cases-slide__hub-scan" />
      </div>
      <span className="use-cases-slide__hub-kicker">CENTRAL SIGNAL</span>
      <strong>Video intelligence</strong>
      <small>Observe · understand · route</small>
    </div>
  )
}

export function UseCasesSlide() {
  return (
    <section
      aria-labelledby="use-cases-slide-title"
      className="use-cases-slide"
    >
      <header className="use-cases-slide__header">
        <div>
          <p>Mission use cases</p>
          <span>04 / applied patterns</span>
        </div>
        <h2 id="use-cases-slide-title">
          One intelligence layer.<br />
          <em>Four ways to act.</em>
        </h2>
        <p>
          Turn visual signals into governed decisions—designed around purpose,
          privacy, and accountable review.
        </p>
      </header>

      <div className="use-cases-slide__stage">
        <div className="use-cases-slide__connections" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>

        <VideoIntelligenceHub />

        <div className="use-cases-slide__stations">
          {useCases.map(
            ({ number, label, title, copy, className, Illustration }) => (
              <article
                className={`use-cases-slide__station ${className}`}
                key={title}
              >
                <div className="use-cases-slide__station-heading">
                  <span>{number}</span>
                  <small>{label}</small>
                </div>
                <Illustration />
                <div className="use-cases-slide__station-copy">
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </div>
              </article>
            ),
          )}
        </div>
      </div>

      <footer className="use-cases-slide__footer">
        <span>VISION → CONTEXT → CONTROLLED ACTION</span>
        <span>Built for mission outcomes, bounded by responsible use</span>
      </footer>
    </section>
  )
}
