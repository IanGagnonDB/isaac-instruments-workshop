import type { OutputDetail } from './api'

export type UiPhase =
  | 'idle'
  | 'ready'
  | 'uploading'
  | 'submitting'
  | 'complete'
  | 'failed'

export interface WorkflowState {
  phase: UiPhase
  sourceName: string
  outputName: string
  elapsedSeconds: number | null
  message: string
  mediaUrl: string
  description: string | null
  descriptionError: string | null
  uploadProgress: number
}

export const initialWorkflowState: WorkflowState = {
  phase: 'idle',
  sourceName: '',
  outputName: '',
  elapsedSeconds: null,
  message: '',
  mediaUrl: '',
  description: null,
  descriptionError: null,
  uploadProgress: 0,
}

export type WorkflowEvent =
  | { type: 'source_selected'; sourceName: string }
  | { type: 'upload_started' }
  | { type: 'upload_progress'; percent: number }
  | { type: 'submitting' }
  | { type: 'output_loaded'; output: OutputDetail }
  | { type: 'failed'; message: string }
  | { type: 'reset' }

export function workflowReducer(
  state: WorkflowState,
  event: WorkflowEvent,
): WorkflowState {
  switch (event.type) {
    case 'source_selected':
      return {
        ...initialWorkflowState,
        phase: event.sourceName ? 'ready' : 'idle',
        sourceName: event.sourceName,
      }
    case 'upload_started':
      return { ...state, phase: 'uploading', message: '', uploadProgress: 0 }
    case 'upload_progress':
      return { ...state, uploadProgress: event.percent }
    case 'submitting':
      return { ...state, phase: 'submitting', message: '' }
    case 'output_loaded':
      return {
        ...initialWorkflowState,
        phase: 'complete',
        outputName: event.output.output_name,
        mediaUrl: event.output.media_url,
        description: event.output.description,
        descriptionError: event.output.description_error ?? null,
        elapsedSeconds: event.output.elapsed_seconds ?? null,
      }
    case 'failed':
      return { ...state, phase: 'failed', message: event.message }
    case 'reset':
      return initialWorkflowState
  }
}
