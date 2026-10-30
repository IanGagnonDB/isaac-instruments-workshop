import {
  type ChangeEvent,
  type FormEvent,
  type ReactNode,
  lazy,
  Suspense,
  useEffect,
  useReducer,
  useRef,
  useState,
} from 'react'
import ReactMarkdown from 'react-markdown'
import databricksLockup from './assets/databricks-lockup.svg'
import fleetImage from './assets/landing/fleet.jpg'
import fleetMask from './assets/landing/fleet-mask.png'
import mooseImage from './assets/landing/moose.jpg'
import mooseMask from './assets/landing/moose-mask.png'
import warfighterImage from './assets/landing/warfighter.jpg'
import warfighterMask from './assets/landing/warfighter-mask.png'
import {
  createTraceId,
  createRun,
  getOutput,
  listInputs,
  listOutputs,
  sendTelemetry,
  streamRun,
  type VideoFile,
  uploadInput,
} from './api'
import {
  initialWorkflowState,
  workflowReducer,
} from './workflow'

const ArchitecturePage = lazy(() =>
  import('./Presentation').then((module) => ({
    default: module.ArchitecturePage,
  })),
)
const CharacteristicsPage = lazy(() =>
  import('./Presentation').then((module) => ({
    default: module.CharacteristicsPage,
  })),
)

const VIDEO_ACCEPT = '.mp4,.mov,.avi,.mkv,.webm,.m4v,.wmv,.flv,.mpeg,.mpg'

function FullscreenIcon({ active }: { active: boolean }) {
  return (
    <svg aria-hidden="true" fill="none" viewBox="0 0 24 24">
      {active ? (
        <path d="M9 4v5H4m11-5v5h5M9 20v-5H4m11 5v-5h5" />
      ) : (
        <path d="M9 4H4v5m11-5h5v5M9 20H4v-5m11 5h5v-5" />
      )}
    </svg>
  )
}

function PresentationStage({ children }: { children: ReactNode }) {
  const stageRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [usesFullscreenFallback, setUsesFullscreenFallback] = useState(false)
  const isMaximized = isFullscreen || usesFullscreenFallback

  useEffect(() => {
    const syncFullscreenState = () => {
      setIsFullscreen(document.fullscreenElement === stageRef.current)
    }
    const exitFallback = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setUsesFullscreenFallback(false)
    }
    document.addEventListener('fullscreenchange', syncFullscreenState)
    document.addEventListener('keydown', exitFallback)
    return () => {
      document.removeEventListener('fullscreenchange', syncFullscreenState)
      document.removeEventListener('keydown', exitFallback)
    }
  }, [])

  async function toggleFullscreen() {
    if (usesFullscreenFallback) {
      setUsesFullscreenFallback(false)
      return
    }
    if (document.fullscreenElement === stageRef.current) {
      await document.exitFullscreen()
      return
    }
    if (!stageRef.current?.requestFullscreen) {
      setUsesFullscreenFallback(true)
      return
    }
    try {
      await stageRef.current.requestFullscreen()
    } catch {
      setUsesFullscreenFallback(true)
    }
  }

  return (
    <div
      className={`presentation-stage ${
        usesFullscreenFallback ? 'is-maximized' : ''
      }`}
      ref={stageRef}
    >
      <button
        aria-label={isMaximized ? 'Exit full screen' : 'View full screen'}
        className="presentation-fullscreen-button"
        onClick={() => void toggleFullscreen()}
        title={isMaximized ? 'Exit full screen' : 'View full screen'}
        type="button"
      >
        <FullscreenIcon active={isMaximized} />
      </button>
      {children}
    </div>
  )
}

type ExperienceSection = 'demo' | 'architecture' | 'characteristics'

function Icon({
  children,
  size = 20,
}: {
  children: ReactNode
  size?: number
}) {
  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {children}
    </svg>
  )
}

function DatabaseIcon() {
  return (
    <Icon>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
      <path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
    </Icon>
  )
}

function UploadIcon() {
  return (
    <Icon size={24}>
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
      <path d="M5 14v5h14v-5" />
    </Icon>
  )
}

function SparkIcon() {
  return (
    <Icon>
      <path d="M12 2l1.7 5.3L19 9l-5.3 1.7L12 16l-1.7-5.3L5 9l5.3-1.7L12 2Z" />
      <path d="M18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z" />
    </Icon>
  )
}

function CheckIcon() {
  return (
    <Icon>
      <path d="m5 12 4 4L19 6" />
    </Icon>
  )
}

function AlertIcon() {
  return (
    <Icon>
      <path d="M12 3 2.8 20h18.4L12 3Z" />
      <path d="M12 9v4m0 3v.01" />
    </Icon>
  )
}

function formatElapsed(seconds: number | null): string {
  if (seconds === null) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainder = seconds % 60
  return hours
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function humanFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function megabytes(bytes: number | null): string {
  return bytes === null ? 'Size unavailable' : `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const LANDING_SCENES = [
  {
    id: 'warfighter',
    title: 'Warfighter',
    prompt: 'soldier',
    image: warfighterImage,
    mask: warfighterMask,
    alt: 'A warfighter walking through a training landscape',
  },
  {
    id: 'fleet',
    title: 'Fleet vehicle',
    prompt: 'white SUV',
    image: fleetImage,
    mask: fleetMask,
    alt: 'A white government fleet vehicle on a civic campus road',
  },
  {
    id: 'moose',
    title: 'Wildlife',
    prompt: 'moose',
    image: mooseImage,
    mask: mooseMask,
    alt: 'A moose standing in a forest clearing',
  },
] as const

function LandingSegmentationVisual() {
  const [sceneIndex, setSceneIndex] = useState(0)
  const scene = LANDING_SCENES[sceneIndex]

  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return
    const interval = window.setInterval(() => {
      setSceneIndex((current) => (current + 1) % LANDING_SCENES.length)
    }, 6800)
    return () => window.clearInterval(interval)
  }, [])

  return (
    <div className="landing-visual">
      <div className="landing-frame">
        <div className="landing-scene" key={scene.id}>
          <img className="landing-scene-image" src={scene.image} alt={scene.alt} />
          <div className="landing-mask-reveal" aria-hidden="true">
            <img src={scene.mask} alt="" />
          </div>
        </div>
        <div className="landing-scene-meta">
          <span>Live segmentation</span>
          <strong>{scene.title}</strong>
          <small>Prompt: “{scene.prompt}”</small>
        </div>
        <div
          className="landing-scan"
          aria-hidden="true"
          key={`scan-${scene.id}`}
        />
        <div className="landing-scene-selector" aria-label="Segmentation examples">
          {LANDING_SCENES.map((item, index) => (
            <button
              aria-label={`Show ${item.title.toLowerCase()} example`}
              aria-pressed={index === sceneIndex}
              key={item.id}
              onClick={() => setSceneIndex(index)}
              type="button"
            >
              <span />
            </button>
          ))}
        </div>
      </div>
      <div className="landing-caption">
        <SparkIcon />
        <span>
          <strong>Prompt-driven segmentation</strong>
          Live, governed, endpoint-backed
        </span>
      </div>
    </div>
  )
}

function App() {
  const [workflow, dispatch] = useReducer(workflowReducer, initialWorkflowState)
  const [launched, setLaunched] = useState(false)
  const [sourceMode, setSourceMode] = useState<'volume' | 'upload'>('volume')
  const [sourceQuery, setSourceQuery] = useState('')
  const [inputOptions, setInputOptions] = useState<VideoFile[]>([])
  const [inputRoot, setInputRoot] = useState('')
  const [sourcePickerOpen, setSourcePickerOpen] = useState(false)
  const [outputQuery, setOutputQuery] = useState('')
  const [outputOptions, setOutputOptions] = useState<VideoFile[]>([])
  const [outputPickerOpen, setOutputPickerOpen] = useState(false)
  const [experienceSection, setExperienceSection] =
    useState<ExperienceSection>('demo')
  const [characteristicSlide, setCharacteristicSlide] = useState(0)
  const [mobileView, setMobileView] = useState<'setup' | 'result'>('setup')
  const [prompt, setPrompt] = useState('')
  const [frameStride, setFrameStride] = useState(15)
  const [truncate, setTruncate] = useState(true)
  const [threshold, setThreshold] = useState(0.5)
  const [outputFps, setOutputFps] = useState<number | null>(8)
  const [transcriptModel, setTranscriptModel] =
    useState<'luna' | 'gemini'>('luna')
  const [deliveryMode, setDeliveryMode] = useState<'final' | 'preview'>('preview')
  const [previewUrl, setPreviewUrl] = useState('')
  const [previewProgress, setPreviewProgress] = useState<{
    processed: number
    total: number
    matched: number
    currentDetections: number
    elapsedSeconds: number
    remainingSeconds: number
  } | null>(null)
  const [processingStatus, setProcessingStatus] = useState('')
  const [hintError, setHintError] = useState('')
  const [lookupError, setLookupError] = useState('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const busy =
    workflow.phase === 'uploading' ||
    workflow.phase === 'submitting'

  useEffect(() => {
    if (!launched) return
    let active = true
    const timer = window.setTimeout(() => {
      listInputs(sourceQuery)
        .then((listing) => {
          if (active) {
            setInputOptions(listing.items)
            setInputRoot(listing.root)
            setHintError('')
          }
        })
        .catch((error: Error) => {
          if (active) setHintError(error.message)
        })
    }, 200)
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [launched, sourceQuery])

  useEffect(() => {
    if (!launched) return
    let active = true
    const timer = window.setTimeout(() => {
      listOutputs(outputQuery)
        .then((listing) => {
          if (active) setOutputOptions(listing.items)
        })
        .catch(() => {
          if (active) setOutputOptions([])
        })
    }, 200)
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [launched, outputQuery])

  function chooseSource(name: string) {
    setSourceQuery(name)
    setSourcePickerOpen(false)
    dispatch({ type: 'source_selected', sourceName: name })
  }

  async function handleStart(event: FormEvent) {
    event.preventDefault()
    if (sourceMode === 'volume' && !workflow.sourceName) {
      dispatch({ type: 'failed', message: 'Select a video before starting.' })
      return
    }
    if (sourceMode === 'upload' && !uploadedFile) {
      dispatch({ type: 'failed', message: 'Choose a video before starting.' })
      return
    }
    if (!prompt.trim()) {
      dispatch({ type: 'failed', message: 'Describe the object to find.' })
      return
    }

    setPreviewUrl('')
    setPreviewProgress(null)
    setProcessingStatus('')
    const traceId = createTraceId()
    sendTelemetry(
      'workflow_started',
      {
        delivery_mode: deliveryMode,
        frame_stride: frameStride,
        threshold,
        truncate,
        output_fps: outputFps,
        transcript_model: transcriptModel,
      },
      traceId,
    )
    let sourceName = workflow.sourceName
    if (sourceMode === 'upload' && uploadedFile) {
      dispatch({ type: 'upload_started' })
      try {
        const upload = await uploadInput(uploadedFile, (percent) => {
          dispatch({ type: 'upload_progress', percent })
        })
        sourceName = upload.source_name
        chooseSource(upload.source_name)
        setSourceMode('volume')
        setUploadedFile(null)
        setInputOptions((items) => [
          { name: upload.source_name, size_bytes: upload.size_bytes },
          ...items,
        ])
      } catch (error) {
        sendTelemetry(
          'workflow_failed',
          {
            delivery_mode: deliveryMode,
            stage: 'upload',
            error_type: error instanceof Error ? error.name : 'UnknownError',
          },
          traceId,
        )
        dispatch({
          type: 'failed',
          message: error instanceof Error ? error.message : 'Upload failed.',
        })
        return
      }
    }

    dispatch({ type: 'submitting' })
    try {
      const input = {
        source_name: sourceName,
        prompt: prompt.trim(),
        frame_stride: frameStride,
        truncate,
        threshold,
        output_fps: outputFps,
        transcript_model: transcriptModel,
      }
      const result =
        deliveryMode === 'preview'
          ? await streamRun(
              input,
              (preview) => {
                setPreviewUrl(`data:image/jpeg;base64,${preview.preview_b64}`)
                setPreviewProgress({
                  processed: preview.processed_frame_count,
                  total: preview.sampled_frame_count,
                  matched: preview.matched_frame_count,
                  currentDetections: preview.current_detection_count,
                  elapsedSeconds: preview.elapsed_seconds,
                  remainingSeconds: preview.estimated_remaining_seconds,
                })
              },
              (status) => setProcessingStatus(status.message),
              traceId,
            )
          : await createRun(input, traceId)
      dispatch({
        type: 'output_loaded',
        output: result,
      })
      setMobileView('result')
      sendTelemetry(
        'workflow_completed',
        {
          delivery_mode: deliveryMode,
          elapsed_seconds: result.elapsed_seconds ?? 0,
          sampled_frames: result.sampled_frame_count ?? 0,
          matched_frames: result.matched_frame_count ?? 0,
        },
        traceId,
      )
    } catch (error) {
      sendTelemetry(
        'workflow_failed',
        {
          delivery_mode: deliveryMode,
          error_type: error instanceof Error ? error.name : 'UnknownError',
        },
        traceId,
      )
      dispatch({
        type: 'failed',
        message:
          error instanceof Error ? error.message : 'Live segmentation failed.',
      })
    }
  }

  async function handleOutputLookup(filename = outputQuery.trim()) {
    if (!filename) return
    setOutputQuery(filename)
    setOutputPickerOpen(false)
    setLookupError('')
    try {
      const output = await getOutput(filename)
      dispatch({ type: 'output_loaded', output })
    } catch (error) {
      setLookupError(
        error instanceof Error ? error.message : 'The output could not be loaded.',
      )
    }
  }

  function reset() {
    dispatch({ type: 'reset' })
    setSourceQuery('')
    setPrompt('')
    setPreviewUrl('')
    setPreviewProgress(null)
    setProcessingStatus('')
    setFrameStride(30)
    setTruncate(true)
    setThreshold(0.5)
    setOutputFps(8)
    setTranscriptModel('luna')
    setLookupError('')
    setMobileView('setup')
  }

  if (!launched) {
    return (
      <div
        className="v3-theme-root landing-shell"
        data-motion="standard"
        data-palette="gsa-hybrid"
        data-scheme="light"
      >
        <header className="topbar landing-topbar">
          <span className="brand">
            <img src={databricksLockup} alt="Databricks" />
          </span>
          <span className="agency-label">Databricks @ The Forge</span>
        </header>
        <main className="landing-main">
          <section className="landing-hero" aria-labelledby="page-title">
            <div className="landing-copy">
              <p className="eyebrow">
                <span />
                Live computer vision
              </p>
              <h1 id="page-title">
                Find what matters <em>when it matters.</em>
              </h1>
              <p>
                Search governed video with natural language. A Databricks
                GPU endpoint segments every match live and returns an explainable
                visual result.
              </p>
              <button
                className="button-primary launch-button"
                onClick={() => setLaunched(true)}
                type="button"
              >
                Launch the demonstration
                <span aria-hidden="true">→</span>
              </button>
            </div>
            <LandingSegmentationVisual />
          </section>
          <div className="landing-proof" aria-label="Workflow characteristics">
            <div>
              <DatabaseIcon />
              <span>
                <strong>Unity Catalog</strong>
                Governed source and output
              </span>
            </div>
            <div>
              <SparkIcon />
              <span>
                <strong>Live Model Serving</strong>
                Synchronous GPU inference
              </span>
            </div>
            <div>
              <CheckIcon />
              <span>
                <strong>Reviewable output</strong>
                Segmented video and description
              </span>
            </div>
          </div>
        </main>
      </div>
    )
  }

  const selectedInput = inputOptions.find(
    (file) => file.name === workflow.sourceName,
  )
  const sectionHeading =
    experienceSection === 'demo'
      ? ['Governed video analysis', 'Find objects in video']
      : experienceSection === 'architecture'
        ? ['Solution blueprint', 'Architecture']
        : ['Platform story', 'Key characteristics']

  return (
    <div
      className="v3-theme-root app-shell cockpit-shell"
      data-motion="standard"
      data-palette="gsa-hybrid"
      data-scheme="light"
    >
      <header className="topbar cockpit-topbar">
        <button
          className="brand brand-button"
          onClick={() => setLaunched(false)}
          type="button"
          aria-label="Return to introduction"
        >
          <img src={databricksLockup} alt="Databricks" />
        </button>
        <div className="cockpit-title">
          <span>Computer vision demonstration</span>
          <strong>Databricks @ The Forge</strong>
        </div>
        <button
          className="button-tertiary intro-button"
          aria-current={experienceSection === 'demo' ? 'page' : undefined}
          onClick={() => {
            setExperienceSection('demo')
            setMobileView('setup')
          }}
          type="button"
        >
          Live Demo
        </button>
      </header>

      <main className="cockpit-main">
        <section className="cockpit-context" aria-labelledby="page-title">
          <div>
            <p className="eyebrow">
              <span />
              {sectionHeading[0]}
            </p>
            <h1 id="page-title">{sectionHeading[1]}</h1>
          </div>
          <nav className="stepper" aria-label="Experience sections">
            {(
              [
                ['demo', 'Live Demo'],
                ['architecture', 'Architecture'],
                ['characteristics', 'Key Characteristics'],
              ] as const
            ).map(([section, label], index) => (
                <button
                  aria-current={
                    experienceSection === section ? 'page' : undefined
                  }
                  className={`step ${
                    experienceSection === section ? 'step-active' : 'step-todo'
                  }`}
                  key={label}
                  onClick={() => setExperienceSection(section)}
                  type="button"
                >
                  <span>{index + 1}</span>
                  <strong>{label}</strong>
                </button>
              ))}
          </nav>
        </section>

        <Suspense
          fallback={
            <section className="presentation-page presentation-loading">
              Loading presentation…
            </section>
          }
        >
          {experienceSection === 'architecture' ? (
            <PresentationStage>
              <ArchitecturePage />
            </PresentationStage>
          ) : experienceSection === 'characteristics' ? (
            <PresentationStage>
              <CharacteristicsPage
                onSlideChange={setCharacteristicSlide}
                slide={characteristicSlide}
              />
            </PresentationStage>
          ) : (
            <>
        <div className="mobile-panel-tabs" aria-label="Application panel">
          <button
            aria-pressed={mobileView === 'setup'}
            className={mobileView === 'setup' ? 'selected' : ''}
            onClick={() => setMobileView('setup')}
            type="button"
          >
            Setup
          </button>
          <button
            aria-pressed={mobileView === 'result'}
            className={mobileView === 'result' ? 'selected' : ''}
            onClick={() => setMobileView('result')}
            type="button"
          >
            Result
            {workflow.phase === 'submitting' && <span className="status-dot" />}
          </button>
        </div>

        <div className="workspace">
          <form
            className={`control-panel panel ${
              mobileView !== 'setup' ? 'mobile-hidden' : ''
            }`}
            onSubmit={handleStart}
          >
            <div className="panel-heading">
              <div>
                <p className="section-label">Analysis setup</p>
                <h2>Configure the search</h2>
              </div>
              {workflow.sourceName && <span className="badge badge-positive">Source ready</span>}
            </div>

            <section className="form-section" aria-labelledby="source-heading">
              <div className="number-label">01</div>
              <div className="form-section-body">
                <h3 id="source-heading">Choose a source video</h3>
                <p>Use a governed video already in Unity Catalog for large files.</p>

                <div className="segmented-control" aria-label="Video source">
                  <button
                    aria-pressed={sourceMode === 'volume'}
                    className={sourceMode === 'volume' ? 'selected' : ''}
                    disabled={busy}
                    onClick={() => setSourceMode('volume')}
                    type="button"
                  >
                    <DatabaseIcon />
                    UC Volume
                  </button>
                  <button
                    aria-pressed={sourceMode === 'upload'}
                    className={sourceMode === 'upload' ? 'selected' : ''}
                    disabled={busy}
                    onClick={() => {
                      setSourceMode('upload')
                      setSourceQuery('')
                      dispatch({ type: 'source_selected', sourceName: '' })
                    }}
                    type="button"
                  >
                    <UploadIcon />
                    Browser upload
                  </button>
                </div>

                {sourceMode === 'volume' ? (
                  <div className="field-group picker-field">
                    <label htmlFor="source-video">
                      Video filename
                      <span className="field-context">Video files only</span>
                    </label>
                    <div className="input-action">
                      <input
                        aria-controls="input-video-options"
                        aria-expanded={sourcePickerOpen}
                        aria-haspopup="listbox"
                        autoComplete="off"
                        disabled={busy}
                        id="source-video"
                        onChange={(event) => {
                          setSourceQuery(event.target.value)
                          setSourcePickerOpen(true)
                          if (event.target.value !== workflow.sourceName) {
                            dispatch({ type: 'source_selected', sourceName: '' })
                          }
                        }}
                        onBlur={() =>
                          window.setTimeout(() => setSourcePickerOpen(false), 120)
                        }
                        onFocus={() => setSourcePickerOpen(true)}
                        placeholder="Search the configured inputs volume"
                        role="combobox"
                        value={sourceQuery}
                      />
                      <button
                        className="button-secondary"
                        disabled={busy || !sourceQuery.trim()}
                        onClick={() => chooseSource(sourceQuery.trim())}
                        type="button"
                      >
                        Select
                      </button>
                    </div>
                    {sourcePickerOpen && (
                      <div
                        className="file-options"
                        id="input-video-options"
                        role="listbox"
                        aria-label="Available videos"
                      >
                        <div className="file-options-heading">
                          <DatabaseIcon />
                          <span title={inputRoot}>{inputRoot || 'Configured inputs Volume'}</span>
                          <strong>{inputOptions.length}</strong>
                        </div>
                        {inputOptions.length ? (
                          inputOptions.map((file) => (
                            <button
                              aria-selected={file.name === workflow.sourceName}
                              key={file.name}
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={() => chooseSource(file.name)}
                              role="option"
                              type="button"
                            >
                              <span className="file-type">VID</span>
                              <span>{file.name}</span>
                              <span className="option-note">
                                {megabytes(file.size_bytes)}
                              </span>
                            </button>
                          ))
                        ) : (
                          <p>No matching videos in this directory.</p>
                        )}
                      </div>
                    )}
                    {hintError && <p className="field-error">{hintError}</p>}
                    {workflow.sourceName && (
                      <p className="selection-confirmation">
                        <CheckIcon />
                        <span>{workflow.sourceName}</span>
                        {selectedInput && (
                          <strong>{megabytes(selectedInput.size_bytes)}</strong>
                        )}
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="upload-area">
                    <input
                      accept={VIDEO_ACCEPT}
                      disabled={busy}
                      id="video-upload"
                      onChange={(event: ChangeEvent<HTMLInputElement>) =>
                        setUploadedFile(event.target.files?.[0] ?? null)
                      }
                      type="file"
                    />
                    <label htmlFor="video-upload">
                      <UploadIcon />
                      <strong>
                        {uploadedFile ? uploadedFile.name : 'Choose a video to upload'}
                      </strong>
                      <span>
                        {uploadedFile
                          ? `${humanFileSize(uploadedFile.size)} · uploads automatically when processing starts`
                          : 'Best for short clips; large sources should use UC Volume'}
                      </span>
                    </label>
                    {workflow.phase === 'uploading' && (
                      <div
                        className="upload-progress"
                        aria-label={`Upload ${workflow.uploadProgress}% complete`}
                      >
                        <span style={{ width: `${workflow.uploadProgress}%` }} />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>

            <section className="form-section" aria-labelledby="prompt-heading">
              <div className="number-label">02</div>
              <div className="form-section-body">
                <h3 id="prompt-heading">Describe what to find</h3>
                <p>Use clear visual language describing an object or activity.</p>
                <div className="field-group">
                  <label htmlFor="prompt">Segmentation prompt</label>
                  <textarea
                    disabled={busy}
                    id="prompt"
                    maxLength={1000}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder="Example: person wearing a red jacket"
                    rows={4}
                    value={prompt}
                  />
                  <div className="prompt-footer">
                    <span>Natural language</span>
                    <span>{prompt.length}/1,000</span>
                  </div>
                </div>

                <div className="field-group">
                  <label>Delivery mode</label>
                  <div className="segmented-control" aria-label="Delivery mode">
                    <button
                      aria-pressed={deliveryMode === 'final'}
                      className={deliveryMode === 'final' ? 'selected' : ''}
                      disabled={busy}
                      onClick={() => setDeliveryMode('final')}
                      type="button"
                    >
                      Final video
                    </button>
                    <button
                      aria-pressed={deliveryMode === 'preview'}
                      className={deliveryMode === 'preview' ? 'selected' : ''}
                      disabled={busy}
                      onClick={() => setDeliveryMode('preview')}
                      type="button"
                    >
                      Live preview
                    </button>
                  </div>
                  <small>
                    Live preview streams sampled frames and shows remaining time as
                    batches finish. Final video waits until the MP4 is ready.
                  </small>
                </div>

                <div className="run-options-grid">
                  <div className="field-group">
                    <label htmlFor="output-fps">Result playback</label>
                    <select
                      disabled={busy}
                      id="output-fps"
                      onChange={(event) =>
                        setOutputFps(
                          event.target.value === 'source'
                            ? null
                            : Number(event.target.value),
                        )
                      }
                      value={outputFps ?? 'source'}
                    >
                      <option value={8}>8 FPS · Quick review</option>
                      <option value={12}>12 FPS · Fast scan</option>
                      <option value={4}>4 FPS · Deliberate</option>
                      <option value="source">Source timing · Preserve duration</option>
                    </select>
                    <small>
                      8 FPS is recommended. Review modes avoid holding sampled frames.
                    </small>
                  </div>
                  <div className="field-group">
                    <label htmlFor="transcript-model">Visual summary model</label>
                    <select
                      disabled={busy}
                      id="transcript-model"
                      onChange={(event) =>
                        setTranscriptModel(
                          event.target.value as 'luna' | 'gemini',
                        )
                      }
                      value={transcriptModel}
                    >
                      <option value="luna">GPT-5.6 Luna</option>
                      <option value="gemini">Gemini 3.8 Flash</option>
                    </select>
                    <small>
                      Luna is the default. Requests use Databricks AI Gateway.
                    </small>
                  </div>
                </div>

                <details className="advanced">
                  <summary>Advanced processing controls</summary>
                  <div className="advanced-grid">
                    <div className="field-group">
                      <label htmlFor="frame-stride">Frame stride</label>
                      <input
                        disabled={busy}
                        id="frame-stride"
                        min={1}
                        onChange={(event) => setFrameStride(Number(event.target.value))}
                        type="number"
                        value={frameStride}
                      />
                      <small>
                        Inspect every Nth source frame. Default is 15. Higher is
                        faster but looks more jumpy. Live preview shows remaining
                        time once the first batch finishes.
                      </small>
                    </div>
                    <div className="field-group">
                      <label htmlFor="threshold">
                        Detection threshold <output>{threshold.toFixed(2)}</output>
                      </label>
                      <input
                        disabled={busy}
                        id="threshold"
                        max={0.9}
                        min={0.25}
                        onChange={(event) => setThreshold(Number(event.target.value))}
                        step={0.05}
                        type="range"
                        value={threshold}
                      />
                      <small>Lower values find more possible matches.</small>
                    </div>
                    <label className="toggle-field">
                      <input
                        checked={truncate}
                        disabled={busy}
                        onChange={(event) => setTruncate(event.target.checked)}
                        type="checkbox"
                      />
                      <span aria-hidden="true" />
                      <span>
                        <strong>Matched segments only</strong>
                        <small>Remove sections with no detected match.</small>
                      </span>
                    </label>
                  </div>
                </details>
              </div>
            </section>

            {workflow.phase === 'failed' && (
              <div className="alert alert-error" role="alert">
                <AlertIcon />
                <div>
                  <strong>We couldn’t complete that request.</strong>
                  <p>{workflow.message}</p>
                </div>
              </div>
            )}

            <div className="form-actions">
              <button
                className="button-primary"
                disabled={
                  busy ||
                  !(sourceMode === 'upload' ? uploadedFile : workflow.sourceName) ||
                  !prompt.trim()
                }
                type="submit"
              >
                <SparkIcon />
                {workflow.phase === 'uploading'
                  ? `Uploading ${workflow.uploadProgress}%…`
                  : workflow.phase === 'submitting'
                    ? 'Processing live…'
                    : 'Find objects in video'}
              </button>
              <button
                className="button-tertiary"
                disabled={workflow.phase === 'submitting'}
                onClick={reset}
                type="button"
              >
                Reset
              </button>
            </div>
          </form>

          <aside
            className={`result-panel panel ${
              mobileView !== 'result' ? 'mobile-hidden' : ''
            }`}
            aria-labelledby="result-heading"
          >
            <div className="panel-heading result-heading">
              <div>
                <p className="section-label">Live result</p>
                <h2 id="result-heading">Review the analysis</h2>
              </div>
              {workflow.phase === 'submitting' && (
                <span className="badge badge-running">
                  <span className="status-dot" />
                  Processing
                </span>
              )}
              {workflow.phase === 'complete' && (
                <span className="badge badge-positive">
                  <CheckIcon />
                  Complete
                </span>
              )}
            </div>

            {workflow.phase === 'submitting' && (
              <div className="run-progress" aria-live="polite">
                <div className="processing-visual">
                  {previewUrl && (
                    <img
                      alt="Latest segmented frame preview"
                      className="live-preview-frame"
                      src={previewUrl}
                    />
                  )}
                  <span className="scan-line" />
                  <div className={`frame-grid ${previewUrl ? 'preview-hidden' : ''}`}>
                    {Array.from({ length: 12 }, (_, index) => (
                      <span key={index} />
                    ))}
                  </div>
                  <div className="processing-copy">
                    <SparkIcon />
                    <strong>
                      {processingStatus
                        ? 'Writing visual description'
                        : 'Analyzing video frames'}
                    </strong>
                    <span>
                      {processingStatus ||
                      (previewProgress
                        ? `${previewProgress.processed} of ${previewProgress.total} sampled frames · target present in ${previewProgress.matched} frames · ${previewProgress.currentDetections} current detection${previewProgress.currentDetections === 1 ? '' : 's'} · ${previewProgress.elapsedSeconds}s elapsed · ~${previewProgress.remainingSeconds}s remaining`
                        : 'The app is calling the warm A10G endpoint directly.')}
                    </span>
                  </div>
                </div>
                <dl className="run-metadata">
                  <div>
                    <dt>Compute</dt>
                    <dd>A10G endpoint</dd>
                  </div>
                  <div>
                    <dt>Elapsed</dt>
                    <dd>
                      {formatElapsed(
                        previewProgress?.elapsedSeconds ?? workflow.elapsedSeconds,
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>Remaining</dt>
                    <dd>
                      {previewProgress
                        ? `~${formatElapsed(previewProgress.remainingSeconds)}`
                        : deliveryMode === 'preview'
                          ? 'Estimating…'
                          : 'Use live preview'}
                    </dd>
                  </div>
                  <div>
                    <dt>State</dt>
                    <dd>PROCESSING</dd>
                  </div>
                </dl>
                {workflow.message && <p className="status-message">{workflow.message}</p>}
              </div>
            )}

            {workflow.phase === 'complete' && workflow.mediaUrl ? (
              <div className="completed-result">
                <div className="video-frame">
                  <video key={workflow.mediaUrl} controls preload="metadata">
                    <source src={workflow.mediaUrl} />
                    Your browser does not support video playback.
                  </video>
                  <span className="video-label">Segmented output</span>
                </div>
                <div className="description-card">
                  <div className="description-title">
                    <SparkIcon />
                    <div>
                      <span>AI-generated description</span>
                      <strong>Verify before using in decisions</strong>
                    </div>
                  </div>
                  <div className="description-markdown">
                    <ReactMarkdown>
                      {workflow.description ||
                        workflow.descriptionError ||
                        'The video is ready, but no description artifact was found.'}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            ) : workflow.phase !== 'submitting' ? (
              <div className="empty-result">
                <div className="empty-illustration" aria-hidden="true">
                  <div className="viewfinder">
                    <span />
                    <span />
                    <span />
                    <span />
                    <div className="target-box" />
                  </div>
                </div>
                <h3>Your analysis will appear here</h3>
                <p>
                  Select a governed source, describe what to find, and start live
                  analysis.
                </p>
              </div>
            ) : null}

            <div className="lookup">
              <div>
                <strong>Open a previous result</strong>
                <span>
                  Select a completed video; its matching AI description loads
                  automatically.
                </span>
              </div>
              <div className="input-action picker-field">
                <input
                  aria-controls="output-video-options"
                  aria-expanded={outputPickerOpen}
                  aria-haspopup="listbox"
                  autoComplete="off"
                  id="output-lookup"
                  onBlur={() =>
                    window.setTimeout(() => setOutputPickerOpen(false), 120)
                  }
                  onChange={(event) => {
                    setOutputQuery(event.target.value)
                    setOutputPickerOpen(true)
                  }}
                  onFocus={() => setOutputPickerOpen(true)}
                  placeholder="Filter completed videos"
                  role="combobox"
                  value={outputQuery}
                />
                <button
                  className="button-secondary"
                  disabled={!outputQuery.trim()}
                  onClick={() => void handleOutputLookup()}
                  type="button"
                >
                  Load
                </button>
                {outputPickerOpen && (
                  <div
                    className="file-options output-options"
                    id="output-video-options"
                    role="listbox"
                    aria-label="Completed video results"
                  >
                    {outputOptions.length ? (
                      outputOptions.map((file) => (
                        <button
                          aria-selected={file.name === outputQuery}
                          key={file.name}
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => void handleOutputLookup(file.name)}
                          role="option"
                          type="button"
                        >
                          <span className="file-type">RESULT</span>
                          <span>{file.name}</span>
                          <span className="option-note">Video + text</span>
                        </button>
                      ))
                    ) : (
                      <p>No matching completed videos.</p>
                    )}
                  </div>
                )}
              </div>
              {lookupError && (
                <p className="field-error" role="alert">
                  {lookupError}
                </p>
              )}
            </div>
          </aside>
        </div>
            </>
          )}
        </Suspense>
      </main>
    </div>
  )
}

export default App
