# Eval set

One photo per item, one row per photo in `items.csv`:

| Column | Meaning |
| --- | --- |
| `image` | photo path, relative to this folder (e.g. `photos/kettle.jpg`) |
| `name` | what the item is, in plain words (e.g. `electric kettle 1.7 L`) |
| `brand` | brand if known, else empty |
| `price_eur` | a real resale price: median of 3 sold or listed ads for the same item |

Run it (costs about a cent per item):

```
uv run off-my-shelf eval data/eval/items.csv --out reports/eval
```

It writes `reports/eval/report.md` (the numbers for the README and tech report) and
`reports/eval/scores.csv` (one row per item). Photos under `data/eval/photos/` are small
JPEGs taken for this project only; keep each under 1 MB.
