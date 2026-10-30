import './mlflow-unity-slide.css'

const modelVersions = [
  { version: 'Version 4', registered: '16:17:46', alias: 'prod' },
  { version: 'Version 3', registered: '11:11:24', alias: '' },
  { version: 'Version 2', registered: '15:55:53', alias: 'challenger' },
  { version: 'Version 1', registered: '17:52:25', alias: 'champion' },
]

const valueStatements = [
  <>
    <strong>Model Registry:</strong> Model lifecycle management with versioning
  </>,
  <>Deploy and organize models with model lineage</>,
  <>Collaboration</>,
  <>Reproducibility</>,
]

export function MlflowUnitySlide() {
  return (
    <section
      className="mlflow-unity-slide"
      aria-labelledby="mlflow-unity-title"
    >
      <header className="mlflow-unity-slide__heading">
        <h2 id="mlflow-unity-title">MLflow + Unity Catalog</h2>
        <p>A centralized model store</p>
      </header>

      <div className="mlflow-unity-slide__body">
        <ul
          className="mlflow-unity-slide__values"
          aria-label="Key model lifecycle capabilities"
        >
          {valueStatements.map((statement, index) => (
            <li key={index}>{statement}</li>
          ))}
        </ul>

        <div
          className="mlflow-unity-slide__product"
          aria-label="Model registry and model lineage examples"
        >
          <section
            className="mlflow-unity-slide__registry"
            aria-labelledby="registry-title"
          >
            <div className="mlflow-unity-slide__crumbs">
              Catalog Explorer <span>/</span> vision_models <span>/</span>{' '}
              inspection_bot
            </div>
            <div className="mlflow-unity-slide__registry-bar">
              <div>
                <span
                  className="mlflow-unity-slide__cube"
                  aria-hidden="true"
                />
                <h3 id="registry-title">inspection_bot</h3>
                <span className="mlflow-unity-slide__star" aria-hidden="true">
                  ☆
                </span>
              </div>
              <span className="mlflow-unity-slide__serve">
                Serve this model
              </span>
            </div>
            <div className="mlflow-unity-slide__tabs" aria-hidden="true">
              <span className="is-active">Overview</span>
              <span>Details</span>
              <span>Permissions</span>
            </div>
            <div className="mlflow-unity-slide__description">
              <span>Description:</span>
              <span>Add description</span>
            </div>
            <div className="mlflow-unity-slide__versions" role="table">
              <div className="mlflow-unity-slide__version-head" role="row">
                <span role="columnheader">Status</span>
                <span role="columnheader">Version</span>
                <span role="columnheader">Time registered</span>
                <span role="columnheader">Aliases</span>
                <span role="columnheader">Registered by</span>
              </div>
              {modelVersions.map((model) => (
                <div
                  className="mlflow-unity-slide__version-row"
                  role="row"
                  key={model.version}
                >
                  <span role="cell">
                    <i aria-label="Ready" role="img" />
                  </span>
                  <span role="cell">{model.version}</span>
                  <span role="cell">2024-04-08 {model.registered}</span>
                  <span role="cell">
                    {model.alias && <em>{model.alias}</em>}
                  </span>
                  <span role="cell">ml.team@databricks.com</span>
                </div>
              ))}
            </div>
          </section>

          <section
            className="mlflow-unity-slide__lineage"
            aria-labelledby="lineage-title"
          >
            <h3 id="lineage-title" className="mlflow-unity-slide__sr-only">
              Data-to-model lineage
            </h3>
            <article className="mlflow-unity-slide__asset">
              <span className="mlflow-unity-slide__asset-type">Table</span>
              <div className="mlflow-unity-slide__asset-name">
                <span aria-hidden="true">▦</span>
                <div>
                  <strong>gold_inspection_features</strong>
                  <small>main.quality.gold_inspection_features</small>
                </div>
              </div>
              <dl>
                <div>
                  <dt>inspection_id</dt>
                  <dd>string</dd>
                </div>
                <div>
                  <dt>captured_at</dt>
                  <dd>timestamp</dd>
                </div>
              </dl>
              <span className="mlflow-unity-slide__more">Show 4 more columns</span>
            </article>

            <div className="mlflow-unity-slide__connector" aria-hidden="true">
              <span>Model version</span>
              <i />
            </div>

            <article className="mlflow-unity-slide__asset mlflow-unity-slide__asset--model">
              <span className="mlflow-unity-slide__asset-type">
                Model version
              </span>
              <div className="mlflow-unity-slide__asset-name">
                <span aria-hidden="true">⌘</span>
                <div>
                  <strong>inspection_bot</strong>
                  <small>ml.team@databricks.com</small>
                </div>
              </div>
              <p>Version 4</p>
              <span className="mlflow-unity-slide__more">1 version</span>
            </article>
          </section>
        </div>
      </div>
    </section>
  )
}
