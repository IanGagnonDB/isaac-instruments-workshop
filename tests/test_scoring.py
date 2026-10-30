import pytest

from video_cls.scoring import scoring_sql, severity


@pytest.mark.parametrize("visible, speed, expected", [
    (False, 30.0, "none"), (True, 0.0, "ok"), (True, 1.9, "ok"), (True, 2.0, "low"),
    (True, 7.9, "low"), (True, 8.0, "medium"), (True, 14.9, "medium"), (True, 15.0, "high"),
])
def test_severity_bands(visible, speed, expected):
    assert severity(visible, speed) == expected


def test_scoring_sql_window_and_bands(spark):
    # 2 Hz traces, 10 s. Sign seen at t=4..5 s; window 1 s -> samples t in [3, 6].
    spark.createDataFrame([
        ("e1", "c1", [30.0] * 6 + [10.0, 9.0, 9.5, 10.0, 11.0, 12.0, 13.0] + [1.0] * 7),   # min near sign 9.0
        ("e2", "c2", [30.0] * 20),                                                      # no sign visible
        ("e3", "c3", [20.0] * 6 + [0.0] * 8 + [20.0] * 6),                              # full stop
    ], "event_id string, clip_id string, speed_kmh array<double>").createOrReplaceTempView("ev")
    spark.createDataFrame([
        ("c1", 4.0, 919, 0.6), ("c1", 5.0, 919, 0.4), ("c1", 8.0, 919, 0.1),           # 8.0 below threshold
        ("c2", 4.0, 919, 0.2), ("c2", 5.0, 3, 0.9),
        ("c3", 4.0, 919, 0.9),
    ], "clip_id string, ts_s double, label int, score double").createOrReplaceTempView("pr")

    rows = {r.event_id: r for r in spark.sql(scoring_sql("ev", "pr", 919, 0.3, 1.0, 2, model_name="m")).collect()}
    assert set(rows) == {"e1", "e2", "e3"}
    assert (rows["e1"].sign_visible, rows["e1"].min_speed_kmh, rows["e1"].severity) == (True, 9.0, "medium")
    assert (rows["e2"].sign_visible, rows["e2"].min_speed_kmh, rows["e2"].severity) == (False, 30.0, "none")
    assert (rows["e3"].sign_visible, rows["e3"].min_speed_kmh, rows["e3"].severity) == (True, 0.0, "ok")
    assert rows["e1"].model == "m"
