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
    streamRun: vi.fn(),
    submitReview: vi.fn(),
  }
})

const summary: api.EventSummary = {
  event_id: 'EVT-0107',
  event_type: 'harsh_acceleration',
  event_label: 'Harsh acceleration (edge trigger)',
  clip: 'nexar_negative_01043.mp4',
  vehicle_id: 'TRK-0233',
  recorded_at: '2026-10-24T10:22:14Z',
  edge_severity: 'low',
  review_status: 'pending',
  reviewed_severity: null,
}

const detail: api.EventDetail = {
  event: {
    event_id: 'EVT-0107',
    simulated: true,
    clip: 'nexar_negative_01043.mp4',
    event_type: 'harsh_acceleration',
    event_label: 'Harsh acceleration (edge trigger)',
    trigger: { time_s: 12 },
    edge_severity: 'low',
    edge_detections: [{ label: 'vehicle', confidence: 0.9, first_seen_s: 0, last_seen_s: 40 }],
    telemetry: {
      speed_kmh: [
        { t_s: 0, speed_kmh: 48 },
        { t_s: 12, speed_kmh: 54 },
        { t_s: 40, speed_kmh: 55 },
      ],
    },
    suggested_prompts: ['vehicle'],
  },
  review_status: 'pending',
  clip_url: '/api/source-media/nexar_negative_01043.mp4',
  output_url: null,
  reviews: [],
  event_types: ['harsh_acceleration', 'no_event', 'other'],
}

describe('EventReviewPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.listEvents).mockResolvedValue({ items: [summary], table: 'c.s.edge_events' })
    vi.mocked(api.getEvent).mockResolvedValue(detail)
  })

  it('runs SAM3 inline on the event clip with the suggested prompt', async () => {
    vi.mocked(api.streamRun).mockResolvedValue({
      output_name: 'nexar_negative_01043.mp4',
      media_url: '/api/media/nexar_negative_01043.mp4',
      description: 'Vehicles move steadily.',
      matched_frame_count: 40,
      sampled_frame_count: 41,
      elapsed_seconds: 30,
    })
    render(<EventReviewPage />)

    expect(await screen.findByDisplayValue('vehicle')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Analyze with SAM3' }))

    await waitFor(() =>
      expect(api.streamRun).toHaveBeenCalledWith(
        expect.objectContaining({ source_name: 'nexar_negative_01043.mp4', prompt: 'vehicle' }),
        expect.any(Function),
        expect.any(Function),
      ),
    )
    expect(await screen.findByText(/found in 40 of 41 sampled frames/)).toBeTruthy()
    expect(screen.getByText('Vehicles move steadily.')).toBeTruthy()
  })

  it('requires a correction and a note when the edge misclassified the event', async () => {
    vi.mocked(api.submitReview).mockResolvedValue({
      event_id: 'EVT-0107',
      severity: 'none',
      edge_classification_correct: false,
      corrected_event_type: 'no_event',
      notes: 'Normal merge.',
      sam3_prompt: null,
      sam3_matched_frames: null,
      sam3_sampled_frames: null,
      reviewer: 'reviewer@example.com',
      submitted_at: '2026-10-30T12:00:00Z',
    })
    render(<EventReviewPage />)
    await screen.findByRole('heading', { name: 'Harsh acceleration (edge trigger)' })
    const submit = screen.getByRole('button', { name: 'Submit grade' }) as HTMLButtonElement

    fireEvent.click(screen.getByRole('radio', { name: /None/ }))
    fireEvent.click(screen.getByRole('radio', { name: 'No, misclassified' }))
    expect(screen.getByRole('radio', { name: 'No, misclassified' }).getAttribute('aria-checked')).toBe('true')
    expect(submit.disabled).toBe(true)

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'no_event' } })
    expect(submit.disabled).toBe(true)
    fireEvent.change(screen.getByPlaceholderText(/What did the edge get wrong/), {
      target: { value: 'Normal merge.' },
    })
    expect(submit.disabled).toBe(false)
    fireEvent.click(submit)

    await waitFor(() =>
      expect(api.submitReview).toHaveBeenCalledWith('EVT-0107', {
        severity: 'none',
        edge_classification_correct: false,
        corrected_event_type: 'no_event',
        notes: 'Normal merge.',
        sam3_prompt: null,
        sam3_matched_frames: null,
        sam3_sampled_frames: null,
      }),
    )
    expect(await screen.findByText('Saved. EVT-0107 is marked reviewed.')).toBeTruthy()
    expect(screen.getByText('Reviewed')).toBeTruthy()
  })

  it('switching back to correct hides the correction field', async () => {
    render(<EventReviewPage />)
    await screen.findByRole('heading', { name: 'Harsh acceleration (edge trigger)' })

    fireEvent.click(screen.getByRole('radio', { name: 'No, misclassified' }))
    expect(screen.queryByRole('combobox')).toBeTruthy()
    fireEvent.click(screen.getByRole('radio', { name: 'Yes, correct' }))
    expect(screen.queryByRole('combobox')).toBeNull()
  })
})
