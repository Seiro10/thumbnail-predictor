# Building a YouTube Thumbnail Scorer with AI — Full Story

> A complete walkthrough of every decision, dead end, and breakthrough — written to help create a YouTube script.

---

## The Idea

The question was simple: **can we use data to predict whether a YouTube thumbnail is good or bad, and give it a score out of 20?**

Not based on gut feeling. Not based on design rules from a blog post. Based on actual performance data from real YouTube channels — what actually got views, and what didn't.

The end result: a machine learning model trained on real channel data, running in the cloud on Google Vertex AI, that takes a thumbnail image and returns a score from 0 to 20.

Here's everything we built, every mistake we made, and every choice we had to make along the way.

---

## Step 1 — Getting the Data

### The approach
We used the YouTube Data API to scrape videos from **18 channels** across two niches: **AI/Tech** and **Business/Finance**. For each video we collected:
- Title, publish date, duration
- View count, like count, comment count
- Subscriber count of the channel at scrape time
- The thumbnail image (downloaded locally)

Total: **~1,243 videos**, thumbnails stored on disk.

### The performance metric: views ÷ subscribers
Raw view count is useless as a metric — a channel with 2M subscribers getting 500k views is actually underperforming, while a 10k-subscriber channel getting 50k views is viral.

So we defined:
```
performance_score = view_count / subscriber_count
```

This normalises for channel size and gives a fair comparison across channels of any size.

### Why 18 channels and not more?
Budget and time. The Vision API costs money per image, and scraping hundreds of channels would take days. 18 channels gave us enough variety to find patterns without breaking the bank.

---

## Step 2 — Labelling the Thumbnails

### The challenge
A machine learning model needs numbers, not images — at this stage. We needed to convert each thumbnail into a set of measurable features: does it have a face? Does it have text? What colours dominate?

### The choice: Google Cloud Vision API
**Option A: Manual labelling** — hire humans, spend weeks, high accuracy but expensive.
**Option B: Computer vision model (self-hosted)** — YOLO for faces, OCR for text, etc. Complex to set up, multiple models to manage.
**Option C: Google Cloud Vision API** — one API call per image, returns faces, text, labels, colours. $1.50 per 1,000 images.

We chose **Option C**. For 1,243 images, the total cost was **~$6–8**. One script, one API, everything in one place.

### What we extracted per thumbnail

| Feature | What it means | Values |
|---|---|---|
| Face present | Is there a human face? | True / False |
| Number of people | How many faces? | 0, 1, 2, 3+ |
| Expression | What emotion? | Neutral, Smile, Surprise, Intense, None |
| Text present | Is there readable text? | True / False |
| Text content | What does it say? | String |
| Dominant colour | What's the main colour palette? | Warm / Cold / Neutral |
| Contrast | How visually distinct are the elements? | Low / Medium / High |
| Background | How busy is the background? | Plain / Busy / Blurred |

All of this saved into `data/auto_labels.csv` and merged with the performance data into `data/videos_enriched.csv`.

---

## Step 3 — The First Correlation Analysis

Before building any model, we asked: **which of these features actually correlate with higher views?**

A basic group-by analysis — comparing average performance_score when a feature is present vs. absent — gave us these multipliers:

| Feature | Performance multiplier |
|---|---|
| Text present | **9.5×** |
| Busy background | **5.5×** |
| Exactly 1 person | **3.4×** |
| Face present | **2.8×** |
| Neutral expression | better than Smile |
| Cold colours | better than Warm |
| High contrast | present in all top-50 videos |

These looked exciting. Text is 9.5× better? Background matters 5.5×? Let's build a scorer.

---

## Step 4 — Five Scoring Attempts (and What Went Wrong)

### v1 — Simple rule-based scorer
**What we did:** Assigned fixed point values based on the multipliers. Text present = 5 pts. Busy background = 4 pts. Face = 3 pts. Etc. Sum = 20.

**The problem:** We double-counted face and number of people — a single person with a face got 6 points for the same thing. Also, the weights were still essentially guesses. We had no way to know if "face = 3 pts" was right vs 4 or 2.

---

### v2 — Ridge Regression
**What we did:** Treated this as a proper machine learning problem. Built a Ridge regression model (a linear model with regularisation) to predict `log(performance_score)` from our features. Scaled the output to 0–20.

**Why Ridge and not plain linear regression?** Our features are correlated — "face present" and "number of people = 1" are almost the same thing. Ridge regularisation handles that gracefully without one feature dominating because of multicollinearity.

**The problem:** The model learned nonsensical weights. High contrast got a **negative** weight despite all top-50 videos having it. Intense expression got the **highest positive** weight, which makes no intuitive sense. The cross-channel confusion was the culprit — more on that below.

---

### v3 — GradientBoosting Classifier
**What we did:** Changed the approach entirely. Instead of predicting a score, we asked: "is this thumbnail a top performer (top 25%) or not?" Then used the predicted probability × 20 as the score.

**Why a classifier?** Classification is often easier than regression on noisy targets. And the GradientBoostingClassifier handles non-linear relationships between features.

**The result:** Cross-validation ROC-AUC = **0.508**. For context, 0.5 is pure chance — a coin flip. The model learned absolutely nothing useful.

**The symptom:** A video with 2.4 million views got a score of **3.0/20**. A low-performing video got **19.8/20**. The scorer was broken.

**Why did it fail?** Two reasons:
1. The dataset (1,243 videos) is too small for GradientBoosting to generalise
2. We were predicting "top performer across all channels" — but being a top performer is mostly about which channel you're on, not your thumbnail

---

### The Root Problem: The Channel Confound

This is the key insight that unlocked everything.

When we computed those multipliers in Step 3 (text=9.5×, busy background=5.5×), we compared videos with text against videos without text **across all 18 channels**.

But channels are not interchangeable. A crypto channel with 500k subscribers naturally has a different views/subscribers ratio than a small tech channel with 20k subscribers. If it happens that channels with text tend to be smaller channels with better ratios, the multiplier is measuring **channel identity**, not **thumbnail quality**.

In other words: we were confusing correlation with causation at the channel level.

---

### v4 — Empirical Multiplier Weights (still confounded)
**What we did:** Used the multipliers (9.5×, 5.5× etc.) to assign weights proportionally. Text got the most points, busy background second, etc.

**The problem:** Same as above — the multipliers were cross-channel averages. Pearson correlation with actual performance: **+0.04**. Basically flat. The scores meant nothing.

---

### v5 — Within-Channel Normalisation (the fix)

**The key insight:** Instead of comparing videos across channels, compare each video **against its own channel's average**.

We Z-scored the performance_score within each channel:
```python
perf_norm = (video_perf - channel_avg) / channel_std
```

Now `perf_norm = +1` means "this video outperformed its channel by 1 standard deviation." This is the real question: **given this channel, did this thumbnail beat the baseline?**

We then recomputed the feature analysis on `perf_norm`:

| Feature | Within-channel diff | Verdict |
|---|---|---|
| Text present | **+0.192** | Strongly positive — real signal |
| 1 person | +0.065 | Positive — real signal |
| Neutral expression | +0.064 | Positive — real signal |
| Cold colours | +0.054 | Positive — real signal |
| **Busy background** | **−0.079** | **Negative — was a confound** |
| **High contrast** | **−0.117** | **Negative — was a confound** |

Busy background and high contrast **flipped negative**. Why? Because channels that tend to use busy, high-contrast thumbnails happen to be channels that generally underperform within their own metric. The thumbnail wasn't causing the performance — it was just a marker of a certain channel style.

This is a classic example of **Simpson's Paradox** — a trend that appears in grouped data but disappears or reverses when you look within subgroups.

**Final v5 score formula:**

| Feature | Points |
|---|---|
| Text (long >50 chars) | 10 pts |
| Text (medium 20–50 chars) | 8.5 pts |
| Text (short <20 chars) | 7 pts |
| No text | 0 pts |
| 1 person | 4 pts |
| 2 people | 2 pts |
| Neutral expression | 3 pts |
| Smile | 2.5 pts |
| Cold colours | 3 pts |
| Neutral colours | 1.5 pts |

**Validation:** Score bands trending correctly — thumbnails scoring 15–20 outperform their channel average, thumbnails scoring 0–5 underperform. Within-channel Pearson correlation: **+0.092**.

---

## Step 5 — Going Beyond Rules: The Neural Network

The rule-based v5 scorer was a solid baseline, but it only used the features the Vision API gave us — coarse labels like "face present" or "background busy." The actual visual content of the thumbnail — composition, lighting, how the face is positioned, how impactful the text looks — was ignored.

To capture that, we needed a model that looks at the actual image.

### The choice: EfficientNetB0 + tabular features

**Option A: Train a CNN from scratch** — needs tens of thousands of images. We have 1,243. This would overfit immediately.

**Option B: Fine-tune all layers of a pretrained model** — still risky with 1,243 samples. The model would memorise the training set.

**Option C: Frozen pretrained backbone + small head** — freeze the pretrained model (no weight updates), just train a small neural network head on top of the extracted features. This is transfer learning done right for small datasets.

**Why EfficientNetB0?**
We considered CLIP (OpenAI's model trained on 400M image-text pairs, which understands text inside images well). CLIP would theoretically be better for thumbnails.

But CLIP requires PyTorch. EfficientNetB0 is built into Keras/TensorFlow. Since we were targeting Vertex AI — which has first-class TensorFlow support and pre-built serving containers — staying in the Keras ecosystem meant:
- No custom Docker image needed
- Simpler deployment pipeline
- Lower cost

**The tradeoff:** EfficientNet was trained on ImageNet (object recognition). CLIP was trained on internet image-text pairs. For thumbnails, CLIP would capture text+image semantics better. EfficientNet still captures visual composition, faces, and colours well. We chose the simpler, cheaper option.

### The architecture

```
Thumbnail image (224×224 pixels)
        │
EfficientNetB0 — frozen, ImageNet weights
        │
GlobalAveragePooling → 1,280-dimensional vector
        │                           │
        │              Vision API features (20 features)
        │              + Channel context (subscriber count,
        │                channel avg performance, niche)
        │                           │
        └─────── Concatenate (1,300 total) ─────────┘
                          │
                   Dense(64, ReLU)
                   Dropout(40%)
                   Dense(32, ReLU)
                   Dropout(30%)
                   Dense(1) → raw score
                          │
                   Scale to 0–20
```

The key design choice: we don't ask EfficientNet to predict the score directly. We use it as a **feature extractor** — it turns the image into a 1,280-number summary of what it sees — and then we combine that with the structured features we already had. A small neural network head learns the final mapping.

### The result

5-fold cross-validation correlation with within-channel normalised performance:
- Mean: **+0.179**
- Per fold: 0.097, 0.339, 0.048, 0.312, 0.102

This is roughly **double** the v5 rule-based scorer (+0.092). The image features genuinely add signal.

The high variance between folds (0.048 to 0.339) reflects the dataset size — 1,243 samples is still small, and results depend heavily on which videos end up in the validation set. More data will reduce this variance.

---

## Step 6 — Deploying to Vertex AI

### Why Vertex AI?
**Option A: Deploy to a VM** — cheapest, but we'd need to manage the server ourselves, handle scaling, updates, uptime.
**Option B: Cloud Run** — serverless containers, pay per request. Great for APIs.
**Option C: Vertex AI** — Google's managed ML platform. Built-in model registry, versioning, batch prediction, monitoring, retraining pipelines. Native TensorFlow support.

We chose **Vertex AI** because it's built for exactly this use case: ML models that need versioning, batch inference, and a retraining loop.

### Why batch prediction, not an online endpoint?

An **online endpoint** stays always-on and responds in milliseconds. It costs ~$0.50–1.50/hour × 24/7 = **$360–1,080/month** even with zero usage.

A **batch prediction job** spins up, runs the predictions, and shuts down. We pay only for the ~10 minutes it runs. Cost: **~$0.05 per batch job**.

For thumbnail scoring, we don't need real-time millisecond responses. We score thumbnails before uploading a video — a 10-minute wait is completely acceptable. Batch is the right choice.

### The deployment pipeline

**Step 1:** Export the trained Keras model as a TF SavedModel with a serving signature that accepts a flat dictionary of all features.

**Step 2:** Upload to Google Cloud Storage (`gs://thumbnail-predictor-models/saved_model/`).

**Step 3:** Register in Vertex AI Model Registry using the **pre-built TF2 serving container** (`europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15`). This avoids building a custom Docker image entirely.

**Step 4:** Inference runs via `inference.py`:
1. Load the thumbnail
2. Run Vision API → extract tabular features
3. Run EfficientNetB0 → extract 1,280-dim embedding
4. Write JSONL to GCS
5. Submit Vertex Batch Prediction job
6. Read results from GCS output

**Region:** `europe-west1`. Always match the region of your GCS bucket and your Vertex resources to avoid egress costs.

### Total cost to deploy

| Item | Cost |
|---|---|
| Vision API (1,243 images, one-time) | ~$8 |
| EfficientNet embedding extraction (local) | $0 |
| Training (local CPU) | $0 |
| GCS storage | < $0.01/month |
| Vertex model registration | $0 |
| Per batch inference job | ~$0.05 |

The entire project cost under **$10** to run end-to-end.

---

## Step 7 — The Feedback Loop (RLHF-style)

Once the model is in production, we want it to improve over time. This is the spirit of RLHF (Reinforcement Learning from Human Feedback) applied to our use case.

### Option A: Automatic loop (no human needed)
After a video is published, real performance data arrives in 30–90 days. That data becomes new training examples. A scheduled pipeline (`vertex/retrain_pipeline.py`) runs monthly:
1. Load new performance data
2. Retrain the model
3. Compare cross-validation correlation — **only promote the new model if it's strictly better**
4. Upload the new version to Vertex Model Registry

This is the simplest and most reliable feedback loop. No human labelling required — the YouTube algorithm itself is the judge.

### Option B: Human preference labelling
Show pairs of thumbnails: "which one do you think will get more views?" Collect these votes, train a **Bradley-Terry reward model** on the pairs, use that as the scoring signal.

This captures human intuition about "what looks clickable" — something that view counts can't fully measure. It's the closest to real RLHF.

**For now:** Option A is implemented and ready to run. Option B is the natural next step once Option A has collected enough retraining cycles.

---

## The Honest Truth About the Limits

The within-channel correlation of +0.179 means the model explains roughly 3% of the variance in video performance. That sounds small — because it is.

But it's not a failure. It reflects reality:

**Thumbnail quality is one factor among many:**
- **Content quality** — a boring video with a great thumbnail will flop
- **Topic timing** — a video about a trending topic outperforms regardless of thumbnail
- **YouTube algorithm** — promotion decisions we have no visibility into
- **Channel authority** — established channels get more impressions per upload
- **Upload time** — day of week, time of day, competing uploads

What the model correctly captures is: **holding everything else equal, thumbnails with text, a single clear face, a neutral expression, and cold-toned colours perform better on average within a channel's own history.**

That's a genuinely useful signal for improving thumbnails before uploading.

---

## Summary of Choices

| Decision | Option chosen | Why | What we rejected |
|---|---|---|---|
| Performance metric | views / subscribers | Normalises for channel size | Raw view count (favours big channels) |
| Thumbnail labelling | Google Vision API | One API, all features, $8 total | Self-hosted YOLO/OCR, manual labelling |
| Correlation analysis | Within-channel z-score | Removes channel confound | Raw cross-channel average (confounded) |
| Image model | EfficientNetB0 frozen | Native Keras, no PyTorch needed | CLIP (better semantics but requires PyTorch) |
| Training data size | 1,243 videos | All available | N/A — would always prefer more |
| Deployment | Vertex AI batch | Pay-per-use, ~$0.05/job | Online endpoint ($360–1,080/month always-on) |
| Serving container | Pre-built TF2 | No Docker build required | Custom container (Cloud Build issue + complexity) |
| Feedback loop | Monthly auto-retrain | Free, uses real data | Human pairwise labelling (more work, more signal) |

---

## Files Reference

| File | What it does |
|---|---|
| `scraper.py` | YouTube API scraping |
| `auto_label_vision.py` | Google Vision API feature extraction |
| `score_thumbnails.py` | Rule-based v5 scorer (0–20) |
| `score_regression.py` | v2 Ridge regression (archived) |
| `score_classifier.py` | v3 GradientBoosting classifier (archived) |
| `vertex/extract_embeddings.py` | EfficientNetB0 embedding extraction |
| `vertex/train.py` | Keras model training + cross-validation |
| `vertex/export_savedmodel.py` | Export + upload TF SavedModel to GCS |
| `vertex/predictor.py` | Prediction logic (batch container) |
| `vertex/retrain_pipeline.py` | Monthly retraining + auto-promotion |
| `inference.py` | End-to-end: new thumbnail → Vertex → score |
| `data/videos_scored_nn.csv` | Dataset scored by the neural model |
| `data/efficientnet_embeddings.npy` | 1,280-dim embeddings for all thumbnails |
| `model/thumbnail_scorer.keras` | Trained Keras model |
| `model/tabular_scaler.pkl` | StandardScaler for tabular features |
| `model/meta.json` | Model metadata + validation metrics |
| `METHODOLOGY.md` | Technical methodology summary |
| `FULL_STORY.md` | This document |
