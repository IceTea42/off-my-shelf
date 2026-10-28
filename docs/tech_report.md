# Off My Shelf: technical report

_Draft. Numbers marked TBD come from the mini eval and real runs (S7)._

## 1. Problem

Most homes hold things nobody uses. Selling them takes one listing per item: photos, a name, a
condition, a fair price, a description. That work, not the selling, is what stops people.
Off My Shelf turns one photo or a short video of a shelf into a listing per item, sorts each
into sell, donate or recycle, and lets an agent act on the whole pack from a typed or spoken goal.

## 2. Pipeline

1. **Stills.** Photos are used as they are; videos give their sharpest keyframes (Laplacian
   variance), so one pass over a shelf is enough.
2. **Inventory (Muse Spark).** One call per still lists the movable items as noun phrases
   (furniture included, fixed room parts and food excluded).
3. **Segmentation (SAM 3.1).** Each phrase is segmented; the most confident mask is kept and
   overlapping detections (IoU > 0.6) are merged.
4. **Listing (Muse Spark).** Each item crop gets one schema-checked JSON listing:
   identification, condition and visible defects, a second-hand price range, title and
   description in English, French or Turkish, and a triage decision.
5. **Product shot.** Built from the photo's own pixels on a studio background. If the mask
   shows the item is partly hidden, Muse Image fills only the hidden (magenta) area; see 3.2.
6. **Pack.** JSON, an HTML page, plain-text listings and a zip.
7. **Agent (Muse Spark tool calls).** Sets pricing modes and triage, drafts buyer replies.

## 3. Design choices

### 3.1 Inventory, then segment
SAM 3.1 needs a phrase; Muse Spark knows what is in the room. Asking Spark for phrases and SAM
for pixels keeps each model on what it does best and gives every item an exact mask.

### 3.2 Exact item, never a lookalike
Early product shots were generated from the crop and drifted: a dark glass vase became a
slipper, dents and dirt were added or removed, extra objects appeared. For a resale listing
that is a failure: the buyer must receive the item in the picture. The final design:

- The default shot uses only real pixels (mask cut-out on white, soft shadow).
- Completion is used only when part of the item is hidden (hull minus mask > 2 % of the item).
- The generated image is **pixel-locked**: every visible pixel is copied back from the photo,
  so only hidden pixels can come from the model.
- **Drift check:** if the model's own version of the visible pixels differs by more than
  0.12 (mean absolute difference), it redrew the item and the shot is rejected.
- **Shot check (Muse Spark):** real crop and completed shot side by side; reject if it is not
  the same item, an object was added, or damage was added or hidden. One corrected retry,
  then fall back to the real-pixel shot. Completed shots are labelled as such.

### 3.3 Structured output that does not break
The API gets a "wire" schema without length constraints and the prompt shows a JSON skeleton;
the reply is fitted and trimmed to the strict schema, with one repair turn on failure.
Failures are saved for inspection.

### 3.4 Cache and provenance
Every model call is cached by a hash of its inputs, and each item records its calls (model,
tokens, cost, latency, cached or not). Re-runs and demos are free and work offline;
`off-my-shelf costs` turns the record into the cost table below.

### 3.5 An agent that can be trusted
Each tool call is checked before it applies: a buyer reply below the item's floor price, or one
moving contact or payment off the marketplace, is rejected and the agent is told why. Replies
are drafts until a person approves them. Spoken goals are transcribed and shown before the
agent acts.

### 3.6 Public demo safety
A spending cap (`OMS_BUDGET_USD`) stops live work on a public deployment once reached; the
bundled demo pack keeps working without any call.

### 3.7 Built for the phone
People declutter standing in front of a shelf, so the web app is an installable Android app
(web app manifest, service worker): camera and video buttons, voice goals, the Android share
sheet into the app (gallery photos start a run) and out of it (a listing with its photo, or a
video ad, to a marketplace or chat app). No app store, one codebase for phone and desktop.

### 3.8 Selling across languages and formats
A listing can be translated with one tap (Muse Spark, text only, cached per language). The agent
can group low-value items into a lot, checked against 70 % of the items' floor prices. Each item
can get a 10-second vertical video ad: its real product shot under a slow zoom, the price and a
hook line written by Muse Spark, encoded locally (PyAV/H.264). Generative video was ruled out:
it needs a GPU and would show an item that is not the one for sale.

## 4. Evaluation

About 30 real items with known resale prices (`data/eval/`). Metrics: identification accuracy,
median absolute price error, share of true prices inside the predicted range, brand
hallucination rate, cost per item. Results: TBD.

## 5. Cost and latency

TBD from `off-my-shelf costs` on a real shelf run. List prices: Muse Spark $1.25 / $4.25 per
million input / output tokens, SAM 3.1 $2.50 per 1,000 images, Muse Image $0.01 per image,
Voice Transcribe $0.18 per hour.

## 6. Limits

- Prices reflect the model's knowledge of the French second-hand market, not live listings.
- Small or heavily hidden items may be missed by the inventory or get a loose mask.
- Completion is skipped (real-pixel shot only) when the checks reject it, so some shots show a
  partly hidden item.

## 7. Next steps

- Live price grounding from marketplace listings.
- Posting to marketplaces through their APIs, still behind human approval.
- Multi-view items: merge sightings of one item across photos into one listing.
