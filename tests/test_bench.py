from video_cls.bench import plan_passes


def test_single_pass_uses_every_slot():
    assert plan_passes("4w", [], n_partitions=48, total_cores=16, max_clips=400) == [
        {"label": "4w", "partitions": 48, "cores": 16, "offset": 0, "limit": 400}]


def test_core_sweep_caps_partitions_and_uses_disjoint_clips():
    passes = plan_passes("sweep", [4, 8, 16], n_partitions=48, total_cores=16, max_clips=48)
    assert [(s["label"], s["partitions"], s["cores"]) for s in passes] == [
        ("sweep_cap4", 4, 4), ("sweep_cap8", 8, 8), ("sweep_cap16", 16, 16)]
    # Slices [0, 48), [48, 96), [96, 144): no pass reads a clip another pass already read.
    assert [(s["offset"], s["limit"]) for s in passes] == [(0, 48), (48, 48), (96, 48)]
