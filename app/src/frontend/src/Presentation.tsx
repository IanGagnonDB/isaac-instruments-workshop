import { type CSSProperties, useEffect } from 'react'
import {
  Background,
  Controls,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
} from '@xyflow/react'
import databricksAppsIcon from './assets/databricks-apps.svg'
import anthropicLogo from './assets/logo-anthropic.svg'
import apacheSparkLogo from './assets/logo-apache-spark.svg'
import azureLogo from './assets/logo-microsoft-azure.svg'
import dspyLogo from './assets/logo-dspy.png'
import haystackLogo from './assets/logo-haystack.svg'
import huggingFaceLogo from './assets/logo-huggingface.svg'
import jaxLogo from './assets/logo-jax.svg'
import langchainLogo from './assets/logo-langchain.svg'
import mosaicMptLogo from './assets/logo-mosaicml-mpt.png'
import openaiLogo from './assets/logo-openai.svg'
import pandasLogo from './assets/logo-pandas.svg'
import pytorchLogo from './assets/logo-pytorch.svg'
import scikitLearnLogo from './assets/logo-scikitlearn.svg'
import statsmodelsLogo from './assets/logo-statsmodels.svg'
import tensorflowLogo from './assets/logo-tensorflow.svg'
import txtaiLogo from './assets/logo-txtai.png'
import xgboostLogo from './assets/logo-xgboost.png'
import modelServingIcon from './assets/model-serving.svg'
import unityCatalogIcon from './assets/unity-catalog.svg'
import { DatabricksAppsSlide } from './slides/DatabricksAppsSlide'
import { DabsSlide } from './slides/DabsSlide'
import { EngineersCornerSlide } from './slides/EngineersCornerSlide'
import { MlflowUnitySlide } from './slides/MlflowUnitySlide'
import { ServerlessGpuSlide } from './slides/ServerlessGpuSlide'
import { UseCasesSlide } from './slides/UseCasesSlide'

function EntryIcon({ type }: { type: 'user' | 'file' }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      viewBox="0 0 24 24"
    >
      {type === 'user' ? (
        <>
          <circle cx="12" cy="8" r="3.5" />
          <path d="M5.5 20c.5-4 2.7-6 6.5-6s6 2 6.5 6" />
        </>
      ) : (
        <>
          <path d="M6 3h8l4 4v14H6Z" />
          <path d="M14 3v5h5M9 14h6m-6 3h4" />
        </>
      )}
    </svg>
  )
}

function ArchitectureLabel({
  icon,
  title,
  detail,
}: {
  icon: string | 'user' | 'file'
  title: string
  detail: string
}) {
  return (
    <div className="architecture-label">
      <span className="architecture-icon">
        {icon === 'user' || icon === 'file' ? (
          <EntryIcon type={icon} />
        ) : (
          <img alt="" src={icon} />
        )}
      </span>
      <span className="architecture-label-copy">
        <strong>{title}</strong>
        <small>{detail}</small>
      </span>
    </div>
  )
}

const architectureNodes: Node[] = [
  {
    id: 'user',
    position: { x: 0, y: 30 },
    sourcePosition: Position.Right,
    type: 'input',
    className: 'architecture-node architecture-node-entry',
    data: {
      label: (
        <ArchitectureLabel
          icon="user"
          title="User"
          detail="Interactive entry"
        />
      ),
    },
  },
  {
    id: 'event',
    position: { x: 0, y: 220 },
    sourcePosition: Position.Right,
    type: 'input',
    className: 'architecture-node architecture-node-entry',
    data: {
      label: (
        <ArchitectureLabel
          icon="file"
          title="File event"
          detail="Automated entry"
        />
      ),
    },
  },
  {
    id: 'app',
    position: { x: 185, y: 30 },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    className: 'architecture-node architecture-node-app',
    data: {
      label: (
        <ArchitectureLabel
          icon={databricksAppsIcon}
          title="Databricks App"
          detail="Upload + launch"
        />
      ),
    },
  },
  {
    id: 'inputs',
    position: { x: 370, y: 125 },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    className: 'architecture-node architecture-node-data',
    data: {
      label: (
        <ArchitectureLabel
          icon={unityCatalogIcon}
          title="UC Volume"
          detail="Governed inputs"
        />
      ),
    },
  },
  {
    id: 'job',
    position: { x: 555, y: 125 },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    className: 'architecture-node architecture-node-job',
    data: {
      label: (
        <ArchitectureLabel
          icon={modelServingIcon}
          title="Warm A10G endpoint"
          detail="Direct invocation"
        />
      ),
    },
  },
  {
    id: 'processing',
    position: { x: 740, y: 125 },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    className: 'architecture-node architecture-node-model',
    data: {
      label: (
        <ArchitectureLabel
          icon={modelServingIcon}
          title="Live segmentation"
          detail="SAM3 frame batches"
        />
      ),
    },
  },
  {
    id: 'result',
    position: { x: 925, y: 125 },
    targetPosition: Position.Left,
    type: 'output',
    className: 'architecture-node architecture-node-result',
    data: {
      label: (
        <ArchitectureLabel
          icon={unityCatalogIcon}
          title="Governed output"
          detail="Video + Markdown → App"
        />
      ),
    },
  },
]

const architectureEdges: Edge[] = [
  {
    id: 'user-app',
    source: 'user',
    target: 'app',
  },
  {
    id: 'event-inputs',
    source: 'event',
    target: 'inputs',
  },
  {
    id: 'app-inputs',
    source: 'app',
    target: 'inputs',
  },
  {
    id: 'inputs-job',
    source: 'inputs',
    target: 'job',
  },
  {
    id: 'job-processing',
    source: 'job',
    target: 'processing',
  },
  {
    id: 'processing-result',
    source: 'processing',
    target: 'result',
  },
].map((edge) => ({
  ...edge,
  type: 'smoothstep',
  markerEnd: { type: MarkerType.ArrowClosed },
  style: { strokeWidth: 2 },
}))

export function ArchitecturePage() {
  return (
    <section className="presentation-page architecture-page">
      <div className="presentation-copy">
        <p className="eyebrow">
          <span />
          Reference architecture
        </p>
        <h2>Live path. Warm GPU. Governed artifacts.</h2>
        <p>
          The app downloads the governed input, samples frames once, invokes
          the warm SAM3 endpoint directly, and publishes the result to the
          governed Volume within the same request.
        </p>
        <div className="architecture-paths">
          <div>
            <strong>01 · Interactive</strong>
            <span>Upload → segment live → review</span>
          </div>
          <div>
            <strong>02 · Oversize fallback</strong>
            <span>Increase stride or use offline batch processing</span>
          </div>
        </div>
      </div>
      <div
        className="architecture-canvas"
        aria-label="Interactive video processing architecture"
      >
        <div className="architecture-stage-labels" aria-hidden="true">
          <span>Entrypoints</span>
          <span>Govern + orchestrate</span>
          <span>Process</span>
          <span>Publish</span>
        </div>
        <ReactFlow
          edges={architectureEdges}
          elementsSelectable={false}
          fitView
          fitViewOptions={{ padding: 0.04 }}
          maxZoom={1.4}
          minZoom={0.35}
          nodes={architectureNodes}
          nodesConnectable={false}
          nodesDraggable={false}
          panOnScroll
          proOptions={{ hideAttribution: true }}
        >
          <Background color="var(--v3-line)" gap={24} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </section>
  )
}

interface CharacteristicSlide {
  source: string
  kicker: string
  title: string
  summary: string
  points: string[]
  tags: string[]
}

const characteristicSlides: CharacteristicSlide[] = [
  {
    source: 'Source slide 02',
    kicker: 'Model choice',
    title: 'Databricks is model agnostic',
    summary:
      'Support for your classic, deep, custom, or LLM-based models',
    points: [
      'Classic and deep learning',
      'Custom functions and APIs',
      'Open and closed LLMs',
      'Chains and autonomous agents',
    ],
    tags: ['CV', 'ML', 'LLMs', 'APIs'],
  },
  {
    source: 'Source slide 03',
    kicker: 'Serverless GPU',
    title: 'Scale AI compute without carrying idle infrastructure.',
    summary:
      'On-demand GPU compute keeps the workflow elastic while Databricks manages the operational layer and critical performance enhancements.',
    points: [
      'Flexible A10 and H100 capacity',
      'Pay only while GPU work is running',
      'Native support for optimized networking',
      'One compute plane for CV, ML, and LLMs',
    ],
    tags: ['On demand', 'Elastic', 'Managed', 'Optimized'],
  },
  {
    source: 'Source slide 04',
    kicker: 'Lifecycle management',
    title: 'Move models from experiment to governed production.',
    summary:
      'MLflow and Unity Catalog create a centralized model system with versioning, lineage, reproducibility, and collaboration built into the lifecycle.',
    points: [
      'Central model registry',
      'Versioned promotion and rollback',
      'Model-to-data lineage',
      'Repeatable team workflows',
    ],
    tags: ['MLflow', 'Unity Catalog', 'Lineage', 'Versions'],
  },
  {
    source: 'Source slide 05',
    kicker: 'Databricks Apps',
    title: 'Build quickly without trading away production confidence.',
    summary:
      'Developers use familiar technologies while the platform provides secure deployment, governed access, and a shorter path from prototype to mission use.',
    points: [
      'Simple developer experience',
      'Production-ready runtime',
      'Built-in data and AI governance',
      'Faster route to user value',
    ],
    tags: ['Simple', 'Secure', 'Governed', 'Ready'],
  },
  {
    source: 'Consolidated source slides 06–10',
    kicker: 'Declarative delivery',
    title: 'Declarative Automation Bundles',
    summary:
      'Declare application resources and environment-specific deployment behavior together.',
    points: [],
    tags: [],
  },
  {
    source: 'Engineer & Architect’s Corner',
    kicker: 'Engineering considerations',
    title: 'Engineer’s Corner',
    summary:
      'Balance frame fidelity, throughput, model limits, governance, and human verification.',
    points: [],
    tags: [],
  },
  {
    source: 'AFS mission patterns',
    kicker: 'Use cases',
    title: 'Mission use cases',
    summary:
      'Apply governed multimodal analysis to validation, occupancy, authorized threat review, and production lines.',
    points: [],
    tags: [],
  },
]

interface ModelBrand {
  name: string
  logos?: string[]
  mark?: string
  wide?: boolean
}

interface ModelFamily {
  title: string
  brands: ModelBrand[]
}

const modelFamilies: ModelFamily[] = [
  {
    title: 'Classic ML',
    brands: [
      { logos: [scikitLearnLogo], name: 'scikit-learn' },
      { logos: [xgboostLogo], name: 'XGBoost', wide: true },
      { logos: [statsmodelsLogo], name: 'statsmodels' },
    ],
  },
  {
    title: 'Deep Learning',
    brands: [
      { logos: [pytorchLogo], name: 'PyTorch' },
      { logos: [tensorflowLogo], name: 'TensorFlow' },
      { logos: [jaxLogo], name: 'JAX', wide: true },
    ],
  },
  {
    title: 'Custom Functions',
    brands: [
      { logos: [apacheSparkLogo], name: 'Apache Spark' },
      { logos: [pandasLogo], name: 'pandas' },
      { mark: '<>', name: 'Custom APIs' },
    ],
  },
  {
    title: 'Closed & Open LLMs',
    brands: [
      { logos: [azureLogo, openaiLogo], name: 'Azure OpenAI' },
      { logos: [mosaicMptLogo], name: 'MosaicML MPT', wide: true },
      { logos: [huggingFaceLogo], name: 'Hugging Face' },
      { logos: [anthropicLogo], name: 'Anthropic' },
    ],
  },
  {
    title: 'Chains & Agents',
    brands: [
      { logos: [langchainLogo], name: 'LangChain' },
      { logos: [dspyLogo], name: 'DSPy', wide: true },
      { logos: [txtaiLogo], name: 'txtai', wide: true },
      { logos: [haystackLogo], name: 'Haystack' },
    ],
  },
]

function ModelAgnosticHero() {
  return (
    <div className="model-agnostic-hero">
      <div className="model-agnostic-heading">
        <div>
          <p className="eyebrow">
            <span />
            Model choice
          </p>
          <p className="slide-source">Source slide 02</p>
        </div>
        <h2>Databricks is model agnostic</h2>
        <p>Support for your classic, deep, custom, or LLM-based models</p>
      </div>
      <div className="model-family-stage">
        <div className="model-family-grid">
          {modelFamilies.map((family) => (
            <article className="model-family-card" key={family.title}>
              <h3>{family.title}</h3>
              <div>
                {family.brands.map((brand) => (
                  <span className="model-brand" key={brand.name}>
                    <span
                      className={`model-brand-mark ${
                        brand.wide ? 'model-brand-mark--wide' : ''
                      }`}
                    >
                      {brand.logos?.map((logo) => (
                        <img alt="" key={logo} src={logo} />
                      )) || brand.mark}
                    </span>
                    <strong>{brand.name}</strong>
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
        <div className="model-platform">
          <span>OPEN · GOVERNED · INTEROPERABLE</span>
          <strong>Pick the best model for your use case</strong>
        </div>
      </div>
    </div>
  )
}

export function CharacteristicsPage({
  slide,
  onSlideChange,
}: {
  slide: number
  onSlideChange: (slide: number) => void
}) {
  const current = characteristicSlides[slide]
  const bespokeSlide =
    slide === 0 ? (
      <ModelAgnosticHero />
    ) : slide === 1 ? (
      <ServerlessGpuSlide />
    ) : slide === 2 ? (
      <MlflowUnitySlide />
    ) : slide === 3 ? (
      <DatabricksAppsSlide />
    ) : slide === 4 ? (
      <DabsSlide />
    ) : slide === 5 ? (
      <EngineersCornerSlide />
    ) : slide === 6 ? (
      <UseCasesSlide />
    ) : null

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft') {
        onSlideChange(Math.max(0, slide - 1))
      }
      if (event.key === 'ArrowRight') {
        onSlideChange(Math.min(characteristicSlides.length - 1, slide + 1))
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onSlideChange, slide])

  return (
    <section
      className={`presentation-page characteristic-page ${
        slide === 0
          ? 'model-agnostic-slide'
          : bespokeSlide
            ? 'bespoke-characteristic-slide'
            : ''
      }`}
      aria-live="polite"
      key={current.source}
    >
      {bespokeSlide ? (
        bespokeSlide
      ) : (
        <>
          <div className="characteristic-copy">
            <div>
              <p className="eyebrow">
                <span />
                {current.kicker}
              </p>
              <p className="slide-source">{current.source}</p>
            </div>
            <h2>{current.title}</h2>
            <p className="characteristic-summary">{current.summary}</p>
            <ul>
              {current.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          </div>
          <div className="characteristic-visual" aria-hidden="true">
            <span className="characteristic-number">
              {String(slide + 1).padStart(2, '0')}
            </span>
            <div className="characteristic-orbit">
              {current.tags.map((tag, index) => (
                <span
                  key={tag}
                  style={{ '--tag-index': index } as CSSProperties}
                >
                  {tag}
                </span>
              ))}
              <strong>LAKEHOUSE</strong>
            </div>
          </div>
        </>
      )}
      <div className="slide-controls">
        <button
          aria-label="Previous characteristic"
          disabled={slide === 0}
          onClick={() => onSlideChange(slide - 1)}
          type="button"
        >
          ←
        </button>
        <div>
          <strong>
            {slide + 1} / {characteristicSlides.length}
          </strong>
          <span>Use arrow keys or controls</span>
        </div>
        <button
          aria-label="Next characteristic"
          disabled={slide === characteristicSlides.length - 1}
          onClick={() => onSlideChange(slide + 1)}
          type="button"
        >
          →
        </button>
      </div>
      <div className="slide-progress" aria-hidden="true">
        {characteristicSlides.map((item, index) => (
          <span className={index === slide ? 'active' : ''} key={item.source} />
        ))}
      </div>
    </section>
  )
}
