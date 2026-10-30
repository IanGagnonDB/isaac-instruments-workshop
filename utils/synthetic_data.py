"""Generate a tiny, explicitly synthetic video dataset. No external downloads."""
import json
import tempfile
import zipfile
from pathlib import Path


def generate_dataset(root):
    import cv2
    import numpy as np
    from utils.download_d2_city import prepare_benchmark

    # Encode on local disk; copy only completed files to the Volume via the existing preparer.
    with tempfile.TemporaryDirectory(prefix='isaac-synthetic-') as directory:
        local = Path(directory)
        videos = []
        for index in range(3):
            path = local / f'synthetic_{index:02d}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (320, 180))
            if not writer.isOpened():
                raise RuntimeError('OpenCV cannot encode MP4; synthetic fallback is unavailable in this environment')
            try:
                for frame_index in range(24):
                    frame = np.zeros((180, 320, 3), dtype=np.uint8)
                    frame[:, :, index] = np.linspace(0, 180, 320, dtype=np.uint8)
                    x = 20 + frame_index * 10
                    cv2.rectangle(frame, (x, 60), (x + 30, 100), (0, 255, 255), -1)
                    cv2.putText(frame, f'SYNTHETIC {index} / {frame_index}', (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    writer.write(frame)
            finally:
                writer.release()
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError('Synthetic video encoding produced no data')
            videos.append(path)
        archive = local / 'synthetic.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            for index, video in enumerate(videos):
                output.write(video, video.name)
                output.writestr(video.with_suffix('.json').name,
                                json.dumps({'event_id': f'synthetic_{index:02d}', 'event_type': 'default',
                                            'synthetic': True}))
        return prepare_benchmark(archive, root=root, copies=1, workers=1)
