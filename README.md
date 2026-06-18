# Part 4 Churn API

## Run

```bash
uvicorn Part4_API.app.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /batch_predict`

Requests can provide either a `customer_id` found in `dataset/rfm_modeling_snapshot.csv` or a complete `features` dictionary with these columns:

```json
[
  "city_tier",
  "age_group",
  "acquisition_channel",
  "loyalty_tier",
  "preferred_category",
  "marketing_consent",
  "recency_days",
  "frequency_180d",
  "monetary_180d",
  "return_rate_180d",
  "avg_discount_pct_180d",
  "avg_rating_180d",
  "category_diversity_180d",
  "ticket_count_90d",
  "negative_ticket_rate_90d",
  "avg_resolution_hours_90d",
  "days_since_signup",
  "sessions_30d",
  "product_views_30d",
  "cart_adds_30d",
  "wishlist_adds_30d",
  "abandoned_carts_30d",
  "email_opens_30d",
  "campaign_clicks_30d",
  "last_visit_days_ago"
]
```

Responses include `churn_probability`, `predicted_class`, `risk_level`, and `risk_explanation`.
