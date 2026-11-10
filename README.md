# ASL fingerspelling recognizer

Recognizes the ASL manual alphabet live from a webcam. MediaPipe finds 21 hand
landmarks per frame, a small PyTorch classifier maps them to a letter, and a
temporal decoder turns the per-frame guesses into typed text.

**Why I built it:** I'm taking an ASL course, and building a recognizer for
the manual alphabet is helping me understand handshapes more precisely.
Fingerspelling is one small part of ASL, used mainly for names and words
without a sign of their own. This project recognizes isolated static
handshapes; it is not sign language recognition or translation.


## Architecture

```
                    OFFLINE                                        LIVE
 ┌──────────────────────┐  ┌────────────────────┐        ┌─────────────────────┐
 │ Public image dataset │  │ Webcam: collect.py │        │ Webcam frame        │
 │ root/<LETTER>/*.jpg  │  │ (your own signing) │        └──────────┬──────────┘
 └──────────┬───────────┘  └─────────┬──────────┘                   │
            │ extract.py             │                              │
            ▼                        ▼                              ▼
 ┌─────────────────────────────────────────────┐        ┌─────────────────────┐
 │ tracker.py  MediaPipe HandLandmarker        │◄───────│ tracker.py (video)  │
 │ image -> 21 landmarks + handedness          │        └──────────┬──────────┘
 └──────────────────────┬──────────────────────┘                   │
                        ▼                                          │
 ┌─────────────────────────────────────────────┐                   │
 │ data.py  one .npz format: raw landmarks,    │                   │
 │ label, signer, source, split group          │                   │
 └───────┬─────────────────────────┬───────────┘                   │
         ▼                         ▼                               ▼
 ┌───────────────┐   ┌──────────────────────────┐       ┌─────────────────────┐
 │ validate.py   │   │ features.py normalize    │◄──────│ features.py         │
 │ QA report +   │   │ splits.py group split    │       │ (same code path)    │
 │ review CSV    │   │ train.py -> runs/x/      │       └──────────┬──────────┘
 └───────────────┘   └────────────┬─────────────┘                  ▼
                                  ▼                      ┌─────────────────────┐
                     ┌──────────────────────────┐       │ model.py Classifier │
                     │ evaluate.py              │       └──────────┬──────────┘
                     │ metrics JSON, confusion  │                  ▼
                     │ matrix, per-signer acc   │       ┌─────────────────────┐
                     └──────────────────────────┘       │ decoder.py smoothing│
                                                        │ live.py overlay/mp4 │
                                                        └─────────────────────┘
```

| File | Role |
|---|---|
| `fingerspell/tracker.py` | MediaPipe wrapper. The only file that imports MediaPipe. |
| `fingerspell/data.py` | The dataset format every tool reads and writes. |
| `fingerspell/extract.py` | Image folders to landmarks, with a log of images where no hand was found. |
| `fingerspell/collect.py` | Record your own labelled samples from the webcam. |
| `fingerspell/validate.py` | Data QA: class balance, detection rate, bad values, outliers to review. |
| `fingerspell/features.py` | Normalization: removes position, size and handedness; keeps orientation. |
| `fingerspell/splits.py` | Group-aware train/val/test split. |
| `fingerspell/model.py` | The MLP, augmentation, and the inference wrapper. |
| `fingerspell/train.py` | Training loop with early stopping. |
| `fingerspell/evaluate.py` | Accuracy, macro-F1, per-class table, top confusions, confusion matrix. |
| `fingerspell/decoder.py` | Frame predictions to committed letters. |
| `fingerspell/live.py` | Webcam demo with on-screen text and video recording. |
| `fingerspell/synthetic.py` | Fake data for testing the pipeline without a camera. |

## Setup (for people who are cloning the repo)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # 9 tests, no camera or dataset needed
```

The MediaPipe hand model (a few MB) downloads itself to `models/` on first use.

## Run it

**1. Get data.** Download the "ASL Alphabet" dataset from Kaggle
(`kaggle datasets download -d grassknoted/asl-alphabet`, about 87,000 images in
29 class folders) and unzip it under `data/raw/`. Check its license on the
Kaggle page before redistributing anything derived from it.

**2. Extract landmarks.** Start with a subsample so you see results in minutes:

```bash
python -m fingerspell.extract --root data/raw/asl_alphabet_train/asl_alphabet_train \
    --out data/processed/kaggle.npz --limit-per-class 600
python -m fingerspell.validate data/processed/kaggle.npz
```

**3. Train and evaluate.**

```bash
python -m fingerspell.train --data data/processed/kaggle.npz --out runs/kaggle
python -m fingerspell.evaluate --run runs/kaggle
```

**4. Record yourself twice: once for training, once for testing.**

    python -m fingerspell.collect --signer me --out data/collected/me_train.npz
    python -m fingerspell.collect --signer me --out data/collected/me_test.npz
    python -m fingerspell.evaluate --run runs/kaggle --data data/collected/me_test.npz --name before

**5. Add your training recording, retrain, and compare on the test recording.**

    python -m fingerspell.train --data data/processed/kaggle.npz data/collected/me_train.npz --out runs/combined
    python -m fingerspell.evaluate --run runs/combined --data data/collected/me_test.npz --name after
    python -m fingerspell.live --run runs/combined --record demo.mp4

## Design decisions

**Landmarks, not pixels.** A 63-number input makes the classifier tiny (about
50k parameters, trains in about a minute on CPU) and indifferent to background,
lighting and skin tone to the extent MediaPipe is. The cost is that anything
MediaPipe gets wrong, the classifier inherits, and landmark quality is known to
be weakest on exactly the hard cases here: closed fists where fingers hide each
other (A, E, M, N, S, T).

**Orientation is kept.** G/Q and K/P are the same handshape in different
orientations, and H is close to U turned sideways. Normalization removes
position, scale and handedness but not rotation, and augmentation rotates by at
most 12 degrees.

**Splits follow groups.** The Kaggle images are consecutive video frames of one
person. A random split puts near-identical frames on both sides and reports
near-perfect accuracy that will not hold up. `splits.py` keeps blocks of 100
consecutive frames together. This still measures the same signer in the same
room, so the number to report is step 4: accuracy on a different person.

**J and Z are motion letters.** They are traced in the air, and a single-frame
classifier only sees a still pose, so J tends to collide with I and Z with D.
The dataset includes still images for both and the model will train on them, but
treat those two as unreliable. `--exclude J Z` at training time drops them; a
proper fix needs a model over landmark sequences.

**The decoder is deliberately simple.** It smooths probabilities and commits a
letter after 8 steady confident frames. It has no language model, and a doubled
letter needs the hand dropped and re-formed, which is not how signers produce
doubles.

## Known limitations

- One signer in the public training data, so expect a large drop on new signers until you add your own.
- Isolated handshapes only. Real fingerspelling is fast and coarticulated: letters blend into each other and are not held.
- One hand, facing the camera. No handling of the signer's viewpoint versus the viewer's.
- Handedness comes from MediaPipe's label, which is sometimes wrong, and a wrong label mirrors the hand.

## Results

Trained on 600 images per class from the Kaggle ASL Alphabet dataset.

| Metric | Value |
|---|---|
| Hand detection rate (MediaPipe) | 79.6% of 16,800 images |
| Test accuracy, dataset's signer | 95.7% (1,910 held-out samples) |
| Macro-F1 | 95.4% |
| End to end (detection × classification) | about 76% |

**Detection varies by letter.** F was found in 96.7% of images and N in only
50.2%. The closed-fist letters M and N are the hardest for MediaPipe because
the fingers hide each other.


**Accuracy on other signers has not been measured yet.** A first spot check on
my own signing suggests recording my own data and retraining is the next step.

**Next:** a held-out test on my own signing, a sequence model for J and Z, and
data from more signers.

## What working on this involved

| Area | Where it lives |
|---|---|
| Data collection | `collect.py`: recording protocol, signer ids, burst grouping |
| Annotating and validating data | `validate.py`: QA report, outlier review list, detection-failure log |
| Training models and running experiments | `train.py`, `evaluate.py`, held-out-signer experiment |
| Building tools and scripts | One shared data format, CLI per stage, tests |

## Verification

Every stage of the pipeline was run end to end:

- **Tests:** all 9 unit tests pass (`python3 -m pytest`).
- **Hand tracking:** MediaPipe's HandLandmarker ran on 16,800 dataset images
  (600 per class) and on live webcam video. It found a hand in 13,366 images
  (79.6%). Detection varied by letter, from 96.7% for F down to 50.2% for N.
- **Data validation:** `validate.py` reported no corrupt or degenerate samples
  and flagged 6.3% of samples as outliers for review.
- **Training:** the classifier reached 96.7% validation accuracy, and two runs
  with the same seed gave identical results. (work in-progress!!!)
- **Evaluation:** 95.7% accuracy and 95.4% macro-F1 on 1,910 held-out samples
  from the dataset's signer. The most common confusions were J→I, P→Q and R→U.
- **Live demo and data collection:** `live.py` and `collect.py` ran on a
  webcam and recorded landmark data. (work in-progress for a demo mp4 soon!!!)

- **How it helps me practice ASL:** the live view draws the tracked hand
  skeleton over my hand and shows which letter the model reads, so I can see
  my handshape from the viewer's side and adjust it as I practice
  fingerspelling.
  
These accuracy figures are for the dataset's single signer, on images where a
hand was detected. Combining detection and classification, roughly 76% of
dataset images are recognized correctly end to end.