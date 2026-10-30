import { type ReactNode } from 'react'

import './dabs-slide.css'

const bundleYaml = `bundle:
  name: video_inspection
workspace:
  host: https://e2-demo.cloud.databricks.com

resources:
  jobs:
    inspect_video:
      name: \${bundle.target}-video-inspection
      tasks:
        - task_key: segment_video
          notebook_task:
            notebook_path: ../src/segment_video.py
          job_cluster_key: gpu_cluster
      job_clusters:
        - job_cluster_key: gpu_cluster
          new_cluster:
            spark_version: 16.4.x-gpu-ml-scala2.12
            node_type_id: g5.4xlarge
            num_workers: 2
  apps:
    inspection_review:
      name: \${bundle.target}-inspection-review
      source_code_path: ../app

targets:
  dev:
    default: true
    mode: development
    workspace:
      root_path: /Workspace/Users/\${workspace.current_user.userName}/.bundle
  prod:
    mode: production
    workspace:
      root_path: /Workspace/Shared/.bundle/\${bundle.name}/\${bundle.target}`

function CodeLine({
  children,
  indent = 0,
  number,
}: {
  children?: ReactNode
  indent?: number
  number: number
}) {
  return (
    <span className={`dabs-code__line dabs-code__line--indent-${indent}`}>
      <span className="dabs-code__number">{number}</span>
      <span className="dabs-code__content">{children}</span>
    </span>
  )
}

export function DabsSlide() {
  return (
    <section className="dabs-slide" aria-labelledby="dabs-slide-title">
      <header className="dabs-slide__header">
        <p>Infrastructure, declared</p>
        <h2 id="dabs-slide-title">
          Declarative Automation Bundles <span>(DABs)</span>
        </h2>
      </header>

      <div className="dabs-slide__body">
        <div className="dabs-concepts" aria-label="Bundle concepts">
          <article className="dabs-concept dabs-concept--identity">
            <span className="dabs-concept__index" aria-hidden="true">
              01
            </span>
            <div>
              <h3>Name and default workspace</h3>
              <p>One deployable project with an explicit workspace boundary.</p>
            </div>
          </article>

          <article className="dabs-concept dabs-concept--resources">
            <span className="dabs-concept__index" aria-hidden="true">
              02
            </span>
            <div>
              <h3>Resource configurations</h3>
              <ul>
                <li>Compute</li>
                <li>Jobs</li>
                <li>Spark Declarative Pipelines</li>
                <li>Models</li>
                <li>Apps</li>
                <li>Lakebase</li>
              </ul>
            </div>
          </article>

          <article className="dabs-concept dabs-concept--targets">
            <span className="dabs-concept__index" aria-hidden="true">
              03
            </span>
            <div>
              <h3>Environment-based specs</h3>
              <p>Control project behavior across development and production.</p>
            </div>
          </article>
        </div>

        <figure className="dabs-editor">
          <figcaption className="dabs-editor__caption">
            <span>Bundle definition</span>
            <strong>One source of truth</strong>
          </figcaption>

          <pre className="dabs-slide__sr-only">
            <code>{bundleYaml}</code>
          </pre>

          <div className="dabs-editor__window" aria-hidden="true">
            <div className="dabs-editor__chrome">
              <span className="dabs-editor__traffic">
                <i />
                <i />
                <i />
              </span>
              <span className="dabs-editor__tab">
                <i />
                databricks.yml
                <b>×</b>
              </span>
              <span className="dabs-editor__actions">⌕ ···</span>
            </div>

            <div className="dabs-editor__breadcrumbs">
              video-inspection <span>›</span> databricks.yml
            </div>

            <div className="dabs-editor__workspace">
              <div className="dabs-editor__rail">
                <span className="is-active">◇</span>
                <span>⌕</span>
                <span>⑂</span>
                <span>▦</span>
              </div>

              <div className="dabs-code">
                <div className="dabs-code__region dabs-code__region--identity">
                  <CodeLine number={1}>
                    <b>bundle</b>:
                  </CodeLine>
                  <CodeLine indent={1} number={2}>
                    <b>name</b>: <em>video_inspection</em>
                  </CodeLine>
                  <CodeLine number={3}>
                    <b>workspace</b>:
                  </CodeLine>
                  <CodeLine indent={1} number={4}>
                    <b>host</b>:{' '}
                    <em>https://e2-demo.cloud.databricks.com</em>
                  </CodeLine>
                </div>

                <CodeLine number={5} />

                <div className="dabs-code__region dabs-code__region--resources">
                  <CodeLine number={6}>
                    <b>resources</b>:
                  </CodeLine>
                  <CodeLine indent={1} number={7}>
                    <b>jobs</b>:
                  </CodeLine>
                  <CodeLine indent={2} number={8}>
                    <b>inspect_video</b>:
                  </CodeLine>
                  <CodeLine indent={3} number={9}>
                    <b>name</b>:{' '}
                    <em>
                      <mark>${'{bundle.target}'}</mark>-video-inspection
                    </em>
                  </CodeLine>
                  <CodeLine indent={3} number={10}>
                    <b>tasks</b>:
                  </CodeLine>
                  <CodeLine indent={4} number={11}>
                    - <b>task_key</b>: <em>segment_video</em>
                  </CodeLine>
                  <CodeLine indent={5} number={12}>
                    <b>notebook_task</b>:
                  </CodeLine>
                  <CodeLine indent={6} number={13}>
                    <b>notebook_path</b>: <em>../src/segment_video.py</em>
                  </CodeLine>
                  <CodeLine indent={5} number={14}>
                    <b>job_cluster_key</b>: <em>gpu_cluster</em>
                  </CodeLine>
                  <CodeLine indent={3} number={15}>
                    <b>job_clusters</b>:
                  </CodeLine>
                  <CodeLine indent={4} number={16}>
                    - <b>job_cluster_key</b>: <em>gpu_cluster</em>
                  </CodeLine>
                  <CodeLine indent={5} number={17}>
                    <b>new_cluster</b>:
                  </CodeLine>
                  <CodeLine indent={6} number={18}>
                    <b>spark_version</b>:{' '}
                    <em>16.4.x-gpu-ml-scala2.12</em>
                  </CodeLine>
                  <CodeLine indent={6} number={19}>
                    <b>node_type_id</b>: <em>g5.4xlarge</em>
                  </CodeLine>
                  <CodeLine indent={6} number={20}>
                    <b>num_workers</b>: <strong>2</strong>
                  </CodeLine>
                  <CodeLine indent={1} number={21}>
                    <b>apps</b>:
                  </CodeLine>
                  <CodeLine indent={2} number={22}>
                    <b>inspection_review</b>:
                  </CodeLine>
                  <CodeLine indent={3} number={23}>
                    <b>source_code_path</b>: <em>../app</em>
                  </CodeLine>
                </div>

                <CodeLine number={24} />

                <div className="dabs-code__region dabs-code__region--targets">
                  <CodeLine number={25}>
                    <b>targets</b>:
                  </CodeLine>
                  <CodeLine indent={1} number={26}>
                    <b>dev</b>:
                  </CodeLine>
                  <CodeLine indent={2} number={27}>
                    <b>default</b>: <strong>true</strong>
                  </CodeLine>
                  <CodeLine indent={2} number={28}>
                    <b>mode</b>: <em>development</em>
                  </CodeLine>
                  <CodeLine indent={2} number={29}>
                    <b>workspace</b>:
                  </CodeLine>
                  <CodeLine indent={3} number={30}>
                    <b>root_path</b>:{' '}
                    <em>/Workspace/Users/$&#123;workspace.current_user.userName&#125;/.bundle</em>
                  </CodeLine>
                  <CodeLine indent={1} number={31}>
                    <b>prod</b>:
                  </CodeLine>
                  <CodeLine indent={2} number={32}>
                    <b>mode</b>: <em>production</em>
                  </CodeLine>
                  <CodeLine indent={2} number={33}>
                    <b>workspace</b>:
                  </CodeLine>
                  <CodeLine indent={3} number={34}>
                    <b>root_path</b>:{' '}
                    <em>/Workspace/Shared/.bundle/$&#123;bundle.name&#125;</em>
                  </CodeLine>
                </div>
              </div>

              <div className="dabs-editor__minimap">
                <i />
                <i />
                <i />
              </div>
            </div>

            <div className="dabs-editor__statusbar">
              <span>main*</span>
              <span>✓ Bundle schema</span>
              <span>Ln 25, Col 1&nbsp;&nbsp; Spaces: 2&nbsp;&nbsp; YAML</span>
            </div>
          </div>
        </figure>
      </div>

      <footer className="dabs-slide__footer">
        <span>CONFIGURATION AS CODE</span>
        <span>Validate · Deploy · Run</span>
      </footer>
    </section>
  )
}
