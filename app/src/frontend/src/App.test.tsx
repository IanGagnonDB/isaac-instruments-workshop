import type { ReactNode } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'

vi.mock('@xyflow/react', () => ({
  Background: () => null,
  Controls: () => null,
  MarkerType: { ArrowClosed: 'arrowclosed' },
  Position: { Left: 'left', Right: 'right' },
  ReactFlow: ({ children }: { children: ReactNode }) => (
    <div data-testid="architecture-flow">{children}</div>
  ),
}))

vi.mock('./api', async () => {
  const actual = await vi.importActual<typeof import('./api')>('./api')
  return {
    ...actual,
    createTraceId: vi.fn(() => 'test-trace-id'),
    sendTelemetry: vi.fn(),
    uploadInput: vi.fn(),
    createRun: vi.fn(),
    streamRun: vi.fn(),
    listInputs: vi.fn().mockResolvedValue({ items: [], root: '/Volumes/test/inputs' }),
    listOutputs: vi.fn().mockResolvedValue({ items: [], root: '/Volumes/test/outputs' }),
  }
})

describe('App', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
    vi.mocked(api.listInputs).mockResolvedValue({
      items: [],
      root: '/Volumes/test/inputs',
    })
    vi.mocked(api.listOutputs).mockResolvedValue({
      items: [],
      root: '/Volumes/test/outputs',
    })
  })

  it('renders the guided setup and accessible empty state', async () => {
    render(<App />)

    expect(
      screen.getByRole('heading', { name: /find what matters/i }),
    ).toBeTruthy()
    expect(
      screen.getByRole('img', {
        name: 'A warfighter walking through a training landscape',
      }),
    ).toBeTruthy()
    fireEvent.click(
      screen.getByRole('button', { name: 'Show fleet vehicle example' }),
    )
    expect(
      screen.getByRole('img', {
        name: 'A white government fleet vehicle on a civic campus road',
      }),
    ).toBeTruthy()
    fireEvent.click(
      screen.getByRole('button', { name: 'Launch the demonstration' }),
    )
    expect(screen.getByText('Databricks @ The Forge')).toBeTruthy()
    expect(
      screen.getByRole('navigation', { name: 'Experience sections' }),
    ).toBeTruthy()
    expect(screen.getByText('Key Characteristics')).toBeTruthy()
    expect(
      (screen.getByLabelText(/Video filename/i) as HTMLInputElement).disabled,
    ).toBe(false)
    expect(
      (screen.getByLabelText('Segmentation prompt') as HTMLTextAreaElement)
        .disabled,
    ).toBe(false)
    const livePreview = screen.getByRole('button', { name: 'Live preview' })
    fireEvent.click(livePreview)
    expect(livePreview.getAttribute('aria-pressed')).toBe('true')
    expect(
      screen.getByRole('heading', { name: 'Your analysis will appear here' }),
    ).toBeTruthy()
  })

  it('shows filtered UC videos in a visible picker', async () => {
    vi.mocked(api.listInputs).mockResolvedValue({
      items: [
        { name: 'intersection.mov', size_bytes: 12_582_912 },
        { name: 'traffic-camera.mp4', size_bytes: 52_428_800 },
      ],
      root: '/Volumes/main/video/auto_segment/inputs',
    })
    render(<App />)
    fireEvent.click(
      screen.getByRole('button', { name: 'Launch the demonstration' }),
    )
    const picker = screen.getByLabelText(/Video filename/i)
    fireEvent.focus(picker)

    await waitFor(() => {
      expect(screen.getByRole('option', { name: /intersection\.mov/i })).toBeTruthy()
    })
    expect(screen.getByText('12.0 MB')).toBeTruthy()
    expect(
      screen.getByTitle('/Volumes/main/video/auto_segment/inputs'),
    ).toBeTruthy()
  })

  it('navigates the architecture and characteristic hero sections', async () => {
    render(<App />)
    fireEvent.click(
      screen.getByRole('button', { name: 'Launch the demonstration' }),
    )

    fireEvent.click(screen.getByRole('button', { name: /Architecture/ }))
    expect(
      await screen.findByRole('heading', {
        name: 'Live path. Warm GPU. Governed artifacts.',
      }),
    ).toBeTruthy()

    fireEvent.click(
      screen.getByRole('button', { name: /Key Characteristics/ }),
    )
    expect(
      await screen.findByRole('heading', {
        name: 'Databricks is model agnostic',
      }),
    ).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Next characteristic' }))
    expect(
      screen.getByRole('heading', {
        name: 'Serverless GPU Compute',
      }),
    ).toBeTruthy()
  })

  it('uploads a browser file automatically before processing', async () => {
    vi.mocked(api.uploadInput).mockResolvedValue({
      source_name: '2026-09-21T1433-tank.mp4',
      size_bytes: 1024,
    })
    vi.mocked(api.streamRun).mockResolvedValue({
      output_name: '2026-09-21T1433-tank.mp4',
      media_url: '/api/media/2026-09-21T1433-tank.mp4',
      description: 'Tank detected.',
      elapsed_seconds: 4,
      sampled_frame_count: 8,
      matched_frame_count: 6,
    })
    render(<App />)
    fireEvent.click(
      screen.getByRole('button', { name: 'Launch the demonstration' }),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Browser upload' }))
    const file = new File(['video'], 'tank.mp4', { type: 'video/mp4' })
    fireEvent.change(screen.getByLabelText(/Choose a video to upload/i), {
      target: { files: [file] },
    })
    fireEvent.change(screen.getByLabelText('Segmentation prompt'), {
      target: { value: 'combat tank' },
    })
    fireEvent.change(screen.getByLabelText('Result playback'), {
      target: { value: 'source' },
    })
    fireEvent.change(screen.getByLabelText('Visual summary model'), {
      target: { value: 'gemini' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Find objects in video' }))

    await waitFor(() =>
      expect(api.uploadInput).toHaveBeenCalledWith(file, expect.any(Function)),
    )
    await waitFor(() =>
      expect(api.streamRun).toHaveBeenCalledWith(
        expect.objectContaining({
          source_name: '2026-09-21T1433-tank.mp4',
          prompt: 'combat tank',
          frame_stride: 15,
          output_fps: null,
          transcript_model: 'gemini',
        }),
        expect.any(Function),
        expect.any(Function),
        'test-trace-id',
      ),
    )
  })

})
