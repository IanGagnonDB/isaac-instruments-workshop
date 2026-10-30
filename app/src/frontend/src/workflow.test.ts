import { describe, expect, it } from 'vitest'
import { initialWorkflowState, workflowReducer } from './workflow'

describe('workflowReducer', () => {
  it('moves from source selection through a completed live result', () => {
    const ready = workflowReducer(initialWorkflowState, {
      type: 'source_selected',
      sourceName: 'traffic.mov',
    })
    const submitting = workflowReducer(ready, { type: 'submitting' })
    const complete = workflowReducer(submitting, {
      type: 'output_loaded',
      output: {
        output_name: 'traffic.mov',
        elapsed_seconds: 26,
        media_url: '/api/media/traffic.mov',
        description: 'A white vehicle crosses the intersection.',
      },
    })

    expect(ready.phase).toBe('ready')
    expect(submitting.phase).toBe('submitting')
    expect(complete).toMatchObject({
      phase: 'complete',
      elapsedSeconds: 26,
      mediaUrl: '/api/media/traffic.mov',
      description: 'A white vehicle crosses the intersection.',
    })
  })

  it('preserves the selected source when live processing fails', () => {
    const ready = workflowReducer(initialWorkflowState, {
      type: 'source_selected',
      sourceName: 'camera.mp4',
    })
    const failed = workflowReducer(ready, {
      type: 'failed',
      message: 'Live processing failed.',
    })
    expect(failed).toMatchObject({
      phase: 'failed',
      sourceName: 'camera.mp4',
      message: 'Live processing failed.',
    })
  })
})
