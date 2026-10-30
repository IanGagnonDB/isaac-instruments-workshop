import './databricks-apps-slide.css'

function SimpleIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="db-apps-slide__illustration"
      viewBox="0 0 120 120"
    >
      <rect className="db-apps-slide__simple-dark" x="8" y="8" width="56" height="56" />
      <rect className="db-apps-slide__simple-mint" x="64" y="8" width="48" height="56" />
      <rect className="db-apps-slide__simple-green" x="8" y="64" width="56" height="48" />
      <rect className="db-apps-slide__simple-teal" x="64" y="64" width="48" height="48" />
      <path
        className="db-apps-slide__simple-shadow"
        d="M23 76 59 54l38 24-37 25Z"
      />
      <circle className="db-apps-slide__simple-disc" cx="61" cy="58" r="27" />
      <path
        className="db-apps-slide__simple-check"
        d="m49 58 8 8 17-20"
      />
    </svg>
  )
}

function ProductionIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="db-apps-slide__illustration"
      viewBox="0 0 120 120"
    >
      <rect className="db-apps-slide__production-base" x="9" y="4" width="102" height="108" />
      <path
        className="db-apps-slide__production-paper"
        d="M9 4h52v13H9zm0 17h52v13H9zm0 17h52v13H9zm0 55h52v19H9zm52-39h50v58H61z"
      />
      <path
        className="db-apps-slide__production-grid"
        d="M9 17h52M9 34h52M9 51h52M9 93h52M61 54v58M86 54v58M61 73h50M61 93h50"
      />
      <path
        className="db-apps-slide__production-frame"
        d="m9 39 52-35 50 35v54l-50 19L9 83Z"
      />
      <path
        className="db-apps-slide__production-cube"
        d="m61 31 22 13v27L61 84 39 71V44Z"
      />
      <path
        className="db-apps-slide__production-cube-lines"
        d="m39 44 22 13 22-13M61 57v27"
      />
    </svg>
  )
}

function GovernanceIllustration() {
  return (
    <svg
      aria-hidden="true"
      className="db-apps-slide__illustration"
      viewBox="0 0 120 120"
    >
      <path
        className="db-apps-slide__governance-coral"
        d="M11 9h98L80 41h29v70H11V41h29Z"
      />
      <path
        className="db-apps-slide__governance-slate"
        d="M11 41h98v40H11Z"
      />
      <path
        className="db-apps-slide__governance-lines"
        d="M11 49h98M11 57h98M11 65h98M11 73h98"
      />
      <path
        className="db-apps-slide__governance-door"
        d="M47 45h26v29H47Z"
      />
      <path
        className="db-apps-slide__governance-arch"
        d="M51 45v-5a9 9 0 0 1 18 0v5"
      />
      <circle className="db-apps-slide__governance-keyhole" cx="60" cy="57" r="3" />
      <path
        className="db-apps-slide__governance-keyhole"
        d="m58.5 59.5-1.5 8h6l-1.5-8"
      />
    </svg>
  )
}

const pillars = [
  {
    title: 'Simple',
    copy:
      'Simplified app development in familiar technologies makes it easy to build without learning another application stack.',
    Illustration: SimpleIllustration,
  },
  {
    title: 'Production-ready',
    copy:
      'A managed foundation gives IT confidence in production readiness, so teams can deploy faster and get to market quickly.',
    Illustration: ProductionIllustration,
  },
  {
    title: 'Built-in governance',
    copy:
      'Data applications are inherently secure and well-governed, leading to greater data security from day one.',
    Illustration: GovernanceIllustration,
  },
]

export function DatabricksAppsSlide() {
  return (
    <section
      aria-labelledby="databricks-apps-slide-title"
      className="db-apps-slide"
    >
      <header className="db-apps-slide__header">
        <h2 id="databricks-apps-slide-title">Introducing Databricks Apps</h2>
        <p>Quickly build secure data applications</p>
      </header>

      <div className="db-apps-slide__pillars">
        {pillars.map(({ title, copy, Illustration }) => (
          <article className="db-apps-slide__pillar" key={title}>
            <div className="db-apps-slide__art">
              <Illustration />
            </div>
            <h3>{title}</h3>
            <p>{copy}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
