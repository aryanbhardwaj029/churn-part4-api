# API Monitoring Plan

## Production Checks

- Availability: `/health` success rate and latency.
- Prediction volume: request counts for `/predict` and `/batch_predict`.
- Input quality: missing feature errors, unknown `customer_id` errors, and schema mismatches.
- Score drift: weekly distribution of `churn_probability`, `risk_level`, and predicted class.
- Data drift: compare live feature medians/category shares against the `2025-09-30` training snapshot.
- Outcome monitoring: once future outcomes mature, track precision, recall, calibration, and segment-level retention lift.

## Alerts

- Health endpoint failure for 5 minutes.
- More than 5% bad requests over 30 minutes.
- Weekly churn score mean shifts by more than 20% relative to training/test baseline.
- Any feature column missing from batch inputs.
