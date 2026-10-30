import './serverless-gpu-slide.css'

const benefits = [
  {
    title: 'Make GPUs easy',
    copy: (
      <>
        Flexible, streamlined compute for training and fine-tuning with{' '}
        <strong>A10 &amp; H100 GPUs</strong>.
      </>
    ),
  },
  {
    title: 'On-demand',
    copy: (
      <>
        Only pay for the GPU you&apos;re using—without carrying idle
        infrastructure.
      </>
    ),
  },
  {
    title: 'Scalable and optimized',
    copy: (
      <>
        Native support for critical GPU enhancements, including{' '}
        <strong>RDMA</strong>.
      </>
    ),
  },
]

const workloads = ['LLMs', 'Image / CV', 'Deep Learning', 'Classic ML']

export function ServerlessGpuSlide() {
  return (
    <section
      className="sg-slide"
      aria-labelledby="serverless-gpu-slide-title"
    >
      <header className="sg-slide__header">
        <p className="sg-slide__eyebrow">AI infrastructure, simplified</p>
        <h2 id="serverless-gpu-slide-title">Serverless GPU Compute</h2>
        <p>Scalable, on-demand compute for AI</p>
      </header>

      <div className="sg-slide__body">
        <div className="sg-slide__benefits" aria-label="Platform benefits">
          {benefits.map((benefit, index) => (
            <article className="sg-benefit" key={benefit.title}>
              <span className="sg-benefit__number" aria-hidden="true">
                {String(index + 1).padStart(2, '0')}
              </span>
              <div>
                <h3>{benefit.title}</h3>
                <p>{benefit.copy}</p>
              </div>
            </article>
          ))}
        </div>

        <div className="sg-slide__platform">
          <ul className="sg-workloads" aria-label="Supported AI workloads">
            {workloads.map((workload, index) => (
              <li key={workload}>
                <span aria-hidden="true">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <strong>{workload}</strong>
              </li>
            ))}
          </ul>

          <div className="sg-platform-banner">
            <span className="sg-platform-banner__signal" aria-hidden="true" />
            <div>
              <strong>Serverless GPU Compute</strong>
              <span>Optimized and managed by Databricks</span>
            </div>
            <span className="sg-platform-banner__status">Ready on demand</span>
          </div>

          <div className="sg-notebook" aria-label="GPU notebook environment">
            <div className="sg-notebook__topbar">
              <div className="sg-notebook__identity">
                <span className="sg-notebook__mark" aria-hidden="true">
                  ▰
                </span>
                <span>
                  <strong>GPU training</strong>
                  <small>Notebook</small>
                </span>
              </div>
              <div className="sg-notebook__actions" aria-hidden="true">
                <span>Python</span>
                <i />
                <i />
                <b>Run all</b>
              </div>
            </div>

            <div className="sg-notebook__workspace">
              <div className="sg-notebook__rail" aria-hidden="true">
                <span>◇</span>
                <span>▱</span>
                <span>⌁</span>
                <span>⚙</span>
              </div>

              <div className="sg-code">
                <div className="sg-code__cell">
                  <span className="sg-code__run" aria-hidden="true">
                    ▶
                  </span>
                  <pre>
                    <code>
                      <span>import</span> torch{'\n'}
                      {'\n'}torch.cuda.is_available()
                    </code>
                  </pre>
                  <span className="sg-code__output">
                    <i aria-hidden="true" />
                    True
                  </span>
                </div>
                <div className="sg-code__metrics" aria-hidden="true">
                  <div>
                    <span>GPU utilization</span>
                    <strong>87%</strong>
                  </div>
                  <i>
                    <b />
                  </i>
                  <small>Accelerator is active</small>
                </div>
              </div>

              <aside className="sg-config" aria-label="GPU configuration">
                <div className="sg-config__heading">
                  <span>
                    <strong>Environment</strong>
                    <small>Compute settings</small>
                  </span>
                  <b aria-hidden="true">×</b>
                </div>
                <div className="sg-config__field">
                  <span>Accelerator</span>
                  <span className="sg-config__select">
                    A10 <b aria-hidden="true">⌄</b>
                  </span>
                </div>
                <div className="sg-config__field">
                  <span>Runtime</span>
                  <span className="sg-config__select">
                    ML Runtime 16.4 LTS <b aria-hidden="true">⌄</b>
                  </span>
                </div>
                <div className="sg-config__meta">
                  <span>
                    <i aria-hidden="true" />
                    Serverless
                  </span>
                  <span>Auto-stop enabled</span>
                </div>
              </aside>
            </div>
          </div>
        </div>
      </div>

      <footer className="sg-slide__footer">
        <span>LAKEHOUSE AI</span>
        <span>GPU capacity when the workload needs it</span>
      </footer>
    </section>
  )
}
