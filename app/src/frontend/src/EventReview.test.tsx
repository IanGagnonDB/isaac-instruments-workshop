import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { EventReviewPage } from './EventReview'
import * as api from './api'

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api')
  return {
    ...actual,
    listEvents: vi.fn(),
    getEvent: vi.fn(),
    submitReview: vi.fn(),
  }
})

const summary: api.EventSummary = {
  event_id: 'EVT-0101',
  event_type: 'stop_sign_violation',
  event_label: 'Rolling stop at stop sign',
  clip: 'nexar_negative_01041.mp4',
  vehicle_id: 'TRK-0412',
  recorded_at: '2026-10-21T14:32:08Z',
  edge_severity: 'medium',
  review_status: 'pending',
  reviewed_severity: null,
}

const detail: api.EventDetail = {
  event: {
    event_id: 'EVT-0101',
    simulated: true,
    clip: 'nexar_negative_01041.mp4',
    event_type: 'stop_sign_violation',
    event_label: 'Rolling stop at stop sign',
    trigger: { time_s: 23.5 },
    edge_severity: 'medium',
    edge_detections: [
      { label: 'stop_sign', confidence: 0.88, first_seen_s: 16.5, last_seen_s: 24.5 },
    ],
    telemetry: {
      speed_kmh: [
        { t_s: 0, speed_kmh: 34 },
        { t_s: 23, speed_kmh: 11 },
        { t_s: 40, speed_kmh: 32 },
      ],
    },
    suggested_prompts: ['stop sign'],
  },
  clip_url: '/api/source-media/nexar_negative_01041.mp4',
  output_url: null,
  reviews: [],
}

describe('EventReviewPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.listEvents).mockResolvedValue({ items: [summary], root: '/Volumes/t/events' })
    vi.mocked(api.getEvent).mockResolvedValue(detail)
  })

  it('submits a severity grade for the selected event', async () => {
    vi.mocked(api.submitReview).mockResolvedValue({
      event_id: 'EVT-0101',
      severity: 'high',
      edge_classification_correct: true,
      notes: 'Rolled through.',
      sam3_prompt: 'stop sign',
      reviewer: 'reviewer@example.com',
      submitted_at: '2026-10-30T12:00:00Z',
    })
    render(<EventReviewPage onAnalyze={vi.fn()} />)

    expect(await screen.findByRole('heading', { name: 'Rolling stop at stop sign' })).toBeTruthy()
    const submit = screen.getByRole('button', { name: 'Submit grade' }) as HTMLButtonElement
    expect(submit.disabled).toBe(true)

    fireEvent.click(screen.getByLabelText(/High/))
    fireEvent.click(screen.getByLabelText('Yes'))
    fireEvent.change(screen.getByPlaceholderText(/What happened/), {
      target: { value: 'Rolled through.' },
    })
    fireEvent.click(submit)

    await waitFor(() =>
      expect(api.submitReview).toHaveBeenCalledWith('EVT-0101', {
        severity: 'high',
        edge_classification_correct: true,
        notes: 'Rolled through.',
        sam3_prompt: 'stop sign',
      }),
    )
    expect(await screen.findByText('Grade submitted by reviewer@example.com.')).toBeTruthy()
  })

  it('sends the clip and a suggested prompt to the live demo', async () => {
    const onAnalyze = vi.fn()
    render(<EventReviewPage onAnalyze={onAnalyze} />)

    fireEvent.click(await screen.findByRole('button', { name: 'Analyze with SAM3' }))

    expect(onAnalyze).toHaveBeenCalledWith('nexar_negative_01041.mp4', 'stop sign')
  })
})
