"""Rolling-stop rule: sign visible (proxy or fine-tuned) + minimum speed near the sign -> severity.

This is a demonstration rule, not a validated severity model.
"""
DEFAULT_THRESHOLDS = {"ok": 2.0, "low": 8.0, "medium": 15.0}


def severity(sign_visible, min_speed_kmh, thresholds=DEFAULT_THRESHOLDS):
    """Python reference of the SQL CASE in scoring_sql(); used by tests."""
    if not sign_visible:
        return "none"
    if min_speed_kmh < thresholds["ok"]:
        return "ok"
    if min_speed_kmh < thresholds["low"]:
        return "low"
    if min_speed_kmh < thresholds["medium"]:
        return "medium"
    return "high"


def scoring_sql(events, predictions, sign_label, score_threshold, window_s, speed_hz,
                thresholds=DEFAULT_THRESHOLDS, model_name="model"):
    """SQL producing event_scores rows.

    predictions: table/view with clip_id, ts_s, label, score (one model/variant only).
    A clip has sign_visible if any frame has label = sign_label and score >= score_threshold.
    min_speed_kmh is taken over [first_visible_ts - window_s, last_visible_ts + window_s];
    for clips without a visible sign it is the whole-clip minimum, shown for context only.
    """
    t = {k: float(v) for k, v in thresholds.items()}
    return f"""
WITH visible AS (
  SELECT clip_id, MIN(ts_s) AS first_ts, MAX(ts_s) AS last_ts
  FROM {predictions}
  WHERE label = {int(sign_label)} AND score >= {float(score_threshold)}
  GROUP BY clip_id
),
speed AS (
  SELECT e.event_id, e.clip_id, pos / {float(speed_hz)} AS t_s, kmh
  FROM {events} e LATERAL VIEW POSEXPLODE(e.speed_kmh) s AS pos, kmh
),
per_clip AS (
  SELECT s.event_id, s.clip_id,
         v.clip_id IS NOT NULL AS sign_visible,
         MIN(CASE WHEN v.clip_id IS NULL
                    OR s.t_s BETWEEN v.first_ts - {float(window_s)} AND v.last_ts + {float(window_s)}
                  THEN s.kmh END) AS min_speed_kmh
  FROM speed s LEFT JOIN visible v ON s.clip_id = v.clip_id
  GROUP BY s.event_id, s.clip_id, v.clip_id
)
SELECT event_id, clip_id, sign_visible, CAST(min_speed_kmh AS DOUBLE) AS min_speed_kmh,
       CASE WHEN NOT sign_visible THEN 'none'
            WHEN min_speed_kmh < {t['ok']} THEN 'ok'
            WHEN min_speed_kmh < {t['low']} THEN 'low'
            WHEN min_speed_kmh < {t['medium']} THEN 'medium'
            ELSE 'high' END AS severity,
       '{model_name}' AS model,
       current_timestamp() AS scored_at
FROM per_clip
"""
