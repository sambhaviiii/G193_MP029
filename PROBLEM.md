# MP029 --- Warehouse Picking Time (Lucknow)

> SCSE3040 MLOps mini project. Read this whole page with your group before
> you write any code. The **Student Handbook** (`STUDENT_HANDBOOK.md`) tells
> you the folder layout, the marks and how to submit. This page tells you
> what is special about *your* problem.

## 1. Your client

**Lucknow Fulfilment Centre** (a made-up organisation).

A fulfilment centre in Lucknow picks thousands of orders a day. Late orders miss the evening truck. The centre wants to predict how long each order will take to pick.

Orders that will be slow can be started first, so they still catch the truck.

## 2. What you must predict

`pick_seconds` --- seconds to pick the whole order, in seconds. It is a number, so this is a
**regression** problem. Measure your models with **MAE** (Mean Absolute
Error): the average size of a mistake, in seconds.

## 3. The files you get

| File | Rows | What it is |
|---|---|---|
| `data/train.csv` | about 1000 | January to June 2026. Your training data. **It is messy.** |
| `data/later.csv` | 300 | July and August 2026. Arrived after your model went live. Clean. |
| `data/holdout_features.csv` | 250 | September 2026. No target. You predict it at the very end. |
| `findings.yaml` | --- | The form you fill in. The marking tool reads it. |

Columns:

| Column | Unit | Meaning |
|---|---|---|
| `record_id` | text | unique id of the record |
| `date` | YYYY-MM-DD | date of the record |
| `items_in_order` | count | items in the order |
| `aisle_distance_m` | metres | walking distance |
| `picker_exp_months` | months | picker's experience |
| `shelf_height_m` | metres | average shelf height |
| `shift_hour` | hour | hour of picking |
| `order_weight_kg` | kg | order weight |
| `zone_type` | text | storage zone (ambient, chilled, bulky) |
| `shift` | text | shift (Morning, Evening, Night) |
| `pick_time_from_scanner` | seconds | pick time read from the hand scanner afterwards |
| `pick_seconds` | seconds | seconds to pick the whole order  **<- the target** |

## 4. The client's warning about `train.csv`

"Our records were typed by many people over six months. Some of the things
that can go wrong with data like ours have gone wrong. We do not know which
ones."

The handbook lists **seven kinds of data problem**. Some of them are hidden
in your `train.csv`, and some are not. For **each of the seven**, find out
whether it is present, and if it is, give the details asked for in
`findings.yaml`. Every answer must come from your own code looking at your
own file: write that code in `src/` and keep it.

Hint: some columns are only filled in *after* the event is over. Ask
yourself, for every column, "would I know this value at the moment I have to
make the prediction?" The holdout file is a good test.

## 5. The client's rules for the prediction service

The request body of `POST /predict` must have exactly these fields:
`items_in_order`, `aisle_distance_m`, `picker_exp_months`, `shelf_height_m`, `shift_hour`, `order_weight_kg`, `zone_type`, `shift`.

The client has two rules. Put them **in the request schema** (pydantic
`Field`), so a bad request gets status 422 and never reaches the model:

1. `shelf_height_m` above 4 is impossible for us. Reject it.
2. `order_weight_kg` below 0 is always a typing mistake. Reject it.

`GET /health` must return `{"status": "ok"}` (you may add more keys).

## 6. Deployment details

| Thing | Must be exactly |
|---|---|
| Docker image name and tag | `mp029-api:1.0` |
| Kubernetes Deployment name | `mp029-api` |
| Kubernetes Service name | `mp029-svc` |
| Replicas | **3** |
| Container port | `8000` |
| Service port | `80` (forwarding to `8000`) |
| Probes | readiness **and** liveness on `/health`; liveness must wait longer |

## 7. Two months later --- monitoring and retraining

Your first model (call it **v1**) was trained on the cleaned `train.csv`
only. Then `later.csv` arrived.

1. For **every numeric column**, compute the **PSI** (Population Stability
   Index) between cleaned `train.csv` and `later.csv` (10 bins, as in P13).
   Which column moved the most?
2. Compute v1's MAE on `later.csv` **separately for each value of
   `shift`**. Which group is now served worst?
3. Sort `later.csv` by date. Retrain **v2** on cleaned `train.csv` plus the
   **first 70%** of `later.csv`. Compare v1 and v2 on the **last 30%**.
4. The client's promotion rule: **promote v2 only if its MAE on that last
   30% is at least 5% lower than v1's.** Record your
   decision. If you promote, v2 becomes the `@champion` in the MLflow
   registry.

## 8. The final prediction

Use your `@champion` model to predict `pick_seconds` for every row of
`data/holdout_features.csv`. Save `predictions/holdout_predictions.csv` with
exactly two columns: `record_id,prediction`. The marking tool knows the true
values and scores your MAE.

## 9. What this problem is NOT

- Not a Kaggle contest. A sensible model that passes the bar is enough; the
  marks are for the **whole pipeline**.
- Not a problem you can solve without running code on your own data. Every
  number in `findings.yaml` is checked against your files.
