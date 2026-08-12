# HireAI — Explainable Candidate Screening

[![CI](https://github.com/abhijeet821/recruitement-automation-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/abhijeet821/recruitement-automation-agent/actions/workflows/ci.yml)

An end-to-end recruitment system that drafts a job description, publishes an
application form, and **ranks applicants against that specific role** with a
score you can take apart and defend.

The screening engine is the substance of the project. It reads a resume into a
structured record, verifies each required skill with a retrieve-then-verify
pipeline, weighs public code where it exists, and produces a 0–100 score
decomposed into weighted dimensions with the evidence behind each one — plus a
separate **confidence** number that says how much was actually known about the
candidate.

Every claim about the scorer is measured. `manage.py evaluate_scorer` runs the
production scorer and three baselines over a labelled set and reports Spearman,
Kendall's τ, NDCG@k, precision/recall@k and MAE.

---

## Results

Measured on a 14-candidate labelled set for a mid-level Python backend role
(`matching/evaluation/golden_set.json`), scored with local `qwen3:14b` +
`bge-m3`:

| Scorer | Spearman ρ | 95% CI | NDCG@5 | 95% CI | MAE |
|---|---|---|---|---|---|
| **ensemble** (production) | **0.964** | [0.86, 0.99] | **0.956** | [0.90, 1.00] | 10.3 |
| `keyword_jd` (JD-aware substrings) | 0.942 | [0.78, 0.99] | 0.716 | [0.67, 1.00] | 10.2 |
| `keyword_legacy` (the original scorer) | 0.822 | [0.46, 0.97] | 0.701 | [0.45, 0.98] | 14.3 |
| `random` | −0.354 | [−0.83, 0.35] | 0.323 | [0.02, 0.72] | 39.7 |

### The honest reading

The ensemble is ahead on every metric. **That gap is not statistically
significant at n=14**, and the confidence intervals are the reason I can say so
rather than guess.

Comparing two scorers by their separate intervals is the wrong test — both are
measured on the *same* candidates, so their errors are correlated. The right one
is a paired bootstrap: resample the candidates, recompute *both* statistics on
that same resample, and look at the difference.

| Comparison | Δ | 95% CI | Won resamples | Significant |
|---|---|---|---|---|
| ensemble − legacy (Spearman) | +0.143 | [−0.030, +0.487] | 93% | no |
| ensemble − legacy (NDCG@5) | +0.255 | [+0.000, +0.554] | 97% | borderline |
| ensemble − keyword_jd (Spearman) | +0.022 | [−0.080, +0.160] | 63% | no |

So the defensible claim is: **the ensemble beat the legacy scorer on 93–97% of
resamples, but 14 candidates cannot establish the difference at 95%
confidence.** Roughly 60–100 labelled candidates would be needed for an effect
this size. Reproduce with `manage.py evaluate_scorer`.

What the sample *does* establish is the qualitative failure mode, which is not a
matter of statistical power. The set contains deliberate traps for literal
matching:

| Candidate | Human label | Legacy keyword score | Ensemble |
|---|---|---|---|
| Node.js backend dev | 3 / 5 | **88.9** (its joint-highest) | 74.5 |
| Senior Java/Spring engineer | 2 / 5 | 66.7 | 53.1 |
| Senior Python/Django engineer | 5 / 5 | 77.8 | 91.2 |

The legacy scorer ranked a Node.js developer above every genuine Python
candidate, because `node`, `api`, `rest`, `sql` and `docker` are all in its
hardcoded list. That is a structural defect in the method, visible in a single
example, and no amount of sample size makes it acceptable.

The app collects recruiter ratings during normal review precisely so this
comparison can be re-run on real labels at a usable sample size — see
[Closing the loop](#closing-the-loop).

---

## Validating the configuration

Two questions the weights leave open, both now measured rather than asserted.

### Does each dimension earn its weight?

`manage.py ablate_scorer` removes each dimension, redistributes its weight, and
re-measures rank agreement. It reuses the persisted scores, so it is arithmetic
over results already computed — instant, rather than the hours a naive re-score
per dimension would take.

```
dimension dropped     weight   spearman   Δ         verdict
(none — full model)   —        0.964      —         baseline
Recruiter rubric      0.19     0.951      -0.013    marginal
Preferred skills      0.05     0.953      -0.011    marginal
Required skills       0.31     0.964      +0.000    no measurable effect
Role alignment        0.13     0.964      +0.000    no measurable effect
Experience level      0.15     0.973      +0.009    no measurable effect
```

**Required skills carries 31% of the weight and removing it moves the ranking by
0.000.** The tempting conclusion — that most of the model is dead weight — is
wrong, and the report says so. Five of seven dimensions are removable because
they are *correlated*: a candidate strong on required skills is also strong on
role alignment and skill quality, so the ordering is over-determined and any one
signal is redundant **given the others**.

Ablation cannot discriminate under those conditions. It needs candidates who are
strong on some dimensions and weak on others, and a 14-row set with cleanly
separated labels does not contain them. The honest output of this experiment is
a better experiment, not a re-tuned model.

### Is the LLM judge stable enough to weight?

The rubric is the only component that can answer the same question differently
twice, and it carries 0.17 of the score. `manage.py rubric_consistency` runs it
repeatedly on one resume:

| Temperature | Overall stdev | Range | Worst-case score swing |
|---|---|---|---|
| 0.1 (production) | 0.000 | 0.000 | **0.0 / 100** |
| 0.8 (8× higher) | 0.009 | 0.020 | **0.3 / 100** |

Under schema-constrained decoding at low temperature the judge is effectively
deterministic, and even at 0.8 the variance moves the final score by a third of
a point. The 0.17 weight is defensible. "Evidence of impact" is the least
reproducible dimension (stdev 0.045) — the one requiring the most judgement,
which is the sensible place for disagreement to show up.

## The finding that shaped the architecture

The natural design for skill matching is "embed both sides, threshold the
cosine". I built that first, then measured it (`manage.py calibrate_similarity`):

```
  1.000  EQUIVALENT  kubernetes <-> k8s
  0.843  EQUIVALENT  python <-> python programming
  0.694  DIFFERENT   postgresql <-> oracle sql
  0.680  DIFFERENT   python <-> java          <-- outranks four TRUE pairs
  0.583  EQUIVALENT  unit testing <-> pytest
  0.546  EQUIVALENT  docker <-> containerisation
  0.463  EQUIVALENT  deep learning <-> pytorch
  0.409  DIFFERENT   django <-> spring boot

  EQUIVALENT  n=10  mean=0.668  min=0.390  max=1.000
  DIFFERENT   n=10  mean=0.518  min=0.409  max=0.694

  OVERLAP of 0.304 — the best possible single threshold reaches only F1=0.700.
```

The classes overlap badly: `python ↔ java` (0.680) scores higher than the
genuine match `docker ↔ containerisation` (0.546). **No threshold separates
them**, and that is not a tuning problem — embeddings encode *topical
relatedness*, not *substitutability*. Python and Java are maximally related;
that is exactly what the model is built to capture. But "has Java" is not
evidence of "knows Python", and that distinction is the whole question.

So skill matching is **retrieve-then-verify**:

1. **Exact match** — normalised string equality → credit 1.0, no model call.
2. **Retrieve** — embeddings shortlist the top-3 plausible resume skills per
   requirement, at a deliberately *low* floor. This stage is tuned for recall.
3. **Verify** — one batched LLM call judges the shortlist as
   `EXACT` / `STRONG` / `PARTIAL` / `NONE`. The model *does* know Java is not
   Python and that PyTorch is a deep-learning framework.

Stage 2 turns "compare against every skill" into "compare against three", which
is what makes stage 3 affordable: one model call per candidate regardless of how
many skills are involved. Verdicts are cached per `(requirement, skill)` pair,
and those repeat heavily across candidates for one role.

Live output for the Java candidate against a Python role:

```
Python             NONE     0.00   -
Django             NONE     0.00   -
REST API design    EXACT    1.00   REST API design
PostgreSQL         PARTIAL  0.50   Oracle SQL
Docker             EXACT    1.00   Docker
unit testing       NONE     0.00   -
```

### The second finding: linear models have no notion of a dealbreaker

With verification working, that Java candidate still scored **67/100 —
"interview"**. Nine years of seniority, strong infrastructure skills and a
degree outvoted the absence of Python, Django *and* testing.

Real screening does not work that way. Hard requirements are **conjunctive**, so
the ensemble applies a multiplicative penalty that ramps from 1.0 at half
coverage down to 0.45 at none. It is a ramp rather than a hard filter so genuine
near-misses degrade smoothly instead of falling off a cliff. That candidate now
scores 53 — "borderline, recruiter review", which is right.

---

## Architecture

```
matching/                    ← pure Python, zero Django imports
├── llm/                     provider abstraction: Ollama | Gemini | fake
│   ├── json_utils.py        recovers JSON from <think> tags, fences, truncation
│   └── cache.py             content-addressed embedding + verdict caches
├── parsing/                 PDF → text → ResumeProfile;  JD → JobSpec
│   └── contacts.py          regex for rigid grammars (email, GitHub handle)
├── enrichment/github.py     public code as evidence
├── features/                → named [0,1] feature vector + provenance
│   └── skills.py            retrieve-then-verify skill matching
├── scoring/                 ensemble, LLM rubric, 3 baselines
├── fairness/                blind-screening redaction, adverse-impact audit
├── evaluation/              metrics, harness, labelled golden set
├── generation/              JD drafting/grading + duration-sized interview guides
└── pipeline.py              the only entry point the web layer imports

hiring_app/                  ← Django
├── models.py                Campaign · Candidate · BackgroundJob · OAuth token
├── views.py                 thin HTTP: parse → call service → report outcome
├── jobs.py                  thread-pool runner with DB-backed progress
├── crypto.py                Fernet encryption for OAuth tokens at rest
└── services/                Google Workspace · LinkedIn · orchestration
```

The `matching` package imports nothing from Django. That boundary is what lets
the same pipeline run from a view, a background job, the evaluation harness, or
a notebook, and it is why the engine is testable without a database.

### Scoring dimensions

Weights sum to 1.0. Any dimension with no evidence is **dropped and its weight
redistributed** — never scored as zero.

| Dimension | Weight | Built from |
|---|---|---|
| Required skills | 0.28 | verified must-have coverage (weighted + hard) |
| Recruiter rubric | 0.17 | LLM judgement on redacted text |
| Experience level | 0.14 | years vs requirement, career progression |
| Role alignment | 0.12 | resume↔JD embedding, title relevance |
| Skill quality | 0.11 | evidence tier (professional > project > listed), recency |
| Code evidence | 0.09 | GitHub substance, language match, recency |
| Preferred skills | 0.05 | nice-to-have coverage |
| Education | 0.04 | stated requirement only |

**Why a hand-weighted linear model and not something learned?** Because there is
no training data yet. A hiring model needs labelled outcomes — who was
interviewed, who performed — and a new deployment has none. A transparent linear
combination is auditable on day one, degrades predictably, and can be explained
to a candidate who asks why they were rejected. `FEATURE_ORDER` is already the
input layer for a learned ranker; these weights become the prior it replaces.

**Why redistribute weight instead of imputing zeros?** Missing evidence is not
negative evidence. A candidate with no GitHub is not worse than one with a
mediocre GitHub — we simply know less. Scoring absence as zero conflates "we
don't know" with "bad" and systematically penalises anyone whose work is
proprietary. Confidence reports how much was known; the score reports quality.
Those are different questions and the UI shows both.

---

## Interview guides

A score tells a recruiter *who* to interview. It does not help them run the
interview. Each candidate page generates questions written from **that
candidate's** résumé, projects and public code, sized to the slot the company
actually books.

**Grounding is the whole feature.** Every question must cite the specific detail
that prompted it, and any question that fails to is discarded before it is shown
— an ungrounded question looks personalised and is not, which is worse than
having none. Real output for a senior Python candidate:

> **You designed idempotent ledger writes on PostgreSQL using advisory locks.
> What trade-offs did you consider versus row-level locks or optimistic
> concurrency control?**
> *From their background:* "Designed idempotent ledger writes on PostgreSQL with
> advisory locks; zero double-charge incidents across 18 months and 2.1M transactions."
> *Listen for:* trade-offs between the mechanisms · performance/correctness
> reasoning · reference to the real-world impact

**Sizing.** The recruiter enters the booked duration; the count follows from it.
Warm-up and the candidate's own questions are reserved, time is budgeted per
category (a project deep-dive genuinely costs more than a skill check), and
categories with no evidence behind them are dropped with their time
redistributed — the same rule the scorer uses for missing dimensions.

| Slot | Questions | Shape |
|---|---|---|
| 15 min | 2 | skill checks only — no room for a deep-dive |
| 30 min | 3 | 1 skill · 1 project · 1 gap probe |
| 45 min | 6 | 2 skill · 1 project · 1 GitHub · 1 gap · 1 experience |
| 60 min | 9 | adds a second project deep-dive and working style |
| 90 min | 14 | 3 project deep-dives, 2 gap probes, more depth |

When the slot is too short for everything, the *least valuable* category is
dropped rather than the most expensive — trimming by cost would strip the
project deep-dive from every short interview, which is exactly backwards.

**Gap probes are a fairness feature, not a gotcha.** Requirements the scorer
could not verify are fed in deliberately, so the interview gives the candidate a
chance to demonstrate a skill the automated step missed, instead of them being
filtered on an inference. The prompt forbids questions about age, family,
nationality, religion, health, marital or visa status, or the personal reasons
behind a career break.

## Fairness

- **Blind screening** (on by default). Name, contact details, address, personal
  URLs, gendered terms and protected-attribute lines are stripped *before* the
  resume reaches the LLM judge. Skills, dates, employers, titles and
  achievements survive intact — redaction that destroys signal is noise, not
  fairness.
- **GitHub absence is never penalised.** People with heavy job commitments,
  proprietary-only work, or caregiving responsibilities have thin public
  profiles for reasons unrelated to competence.
- **Inclusive-language grading** on generated JDs, flagging masculine-coded,
  age-coded, ableist and exclusionary wording with suggested alternatives
  (Gaucher, Friesen & Kay, 2011).
- **Adverse-impact auditing** — `manage.py audit_fairness` computes selection
  rates, four-fifths impact ratios and per-group score distributions.

Group labels must come from **voluntary self-identification** and are passed in
as a file. The system will not infer demographics from names or resumes:
name-based ethnicity inference is inaccurate and is itself a discriminatory act,
so doing it to "check for bias" introduces the exact harm it claims to detect.

---

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com).

```bash
ollama serve
ollama pull qwen3:14b      # reasoning: extraction, verification, rubric
ollama pull bge-m3         # embeddings (1024-d)

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env       # defaults work for local development
python manage.py migrate
python manage.py doctor    # verifies every dependency, tells you what's missing
python manage.py runserver
```

`llama3.2:3b` works for a faster, weaker setup (`OLLAMA_MODEL=llama3.2:3b`).
For hosted inference: `pip install -r requirements-gemini.txt`, then set
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY`.

Google Workspace features (forms, sheets, email) need OAuth credentials — put
`client_secrets.json` in the project root, or set `GOOGLE_CLIENT_SECRETS`.
**Scoring works fully without them.**

### Seeing it work without Google

Applicants normally arrive through a Google Form, which makes the screening half
awkward to demo. This loads the 14 labelled candidates directly:

```bash
python manage.py seed_demo          # ~7 min: runs the real pipeline on each CV
python manage.py seed_demo --fast   # ~2 min: skips the LLM rubric
```

You get a scored campaign with the candidate table, per-dimension score
breakdowns, evidence panels and the recruiter-rating flow — and because the
golden-set labels are seeded as ratings, `evaluate_scorer --campaign <id>`
benchmarks the scorer against them straight away. Seeded ratings are marked
synthetic in the UI so they cannot be mistaken for real recruiter judgements.

### Commands

| Command | Purpose |
|---|---|
| `manage.py doctor` | Check every dependency; says exactly what to fix |
| `manage.py seed_demo` | Load the labelled set as a scored campaign — the whole screening UI, with no Google setup |
| `manage.py evaluate_scorer` | Benchmark against all baselines, with bootstrap CIs and a paired significance test |
| `manage.py evaluate_scorer --campaign N` | Benchmark against real recruiter ratings |
| `manage.py ablate_scorer --campaign N` | Measure what each scoring dimension contributes |
| `manage.py rubric_consistency --candidate N` | Measure run-to-run variance of the LLM judge |
| `manage.py calibrate_similarity` | Re-measure embedding separation |
| `manage.py audit_fairness --campaign N` | Adverse-impact report |
| `pytest` | 245 tests, no model server required |

---

## Closing the loop

The synthetic golden set validates the *pipeline*. It cannot validate the
*weights* — only real hiring judgement can.

So every candidate page has a 0–5 recruiter rating field. Ratings accumulate as
a by-product of normal review, and then:

```bash
python manage.py evaluate_scorer --campaign 3
```

builds an evaluation set from that campaign's real applicants and real labels,
and reports how well the scorer agrees with the recruiter who used it. That is
the path from "this is a plausible heuristic" to "this is measured on our data",
and it is deliberately the shortest path in the product.

---

## What changed from v1

The first version was a competent Django + Google APIs integration whose "AI
scoring" was this:

```python
def _score_text(self, text):
    keywords = ["python", "django", "api", "sql", "rest", "docker", "java", "node", "aws"]
    return sum(1 for k in keywords if k in text.lower())
```

Nine hardcoded terms, independent of the role being hired for. It is preserved
verbatim as `KeywordBaseline` — deleted from production, kept as the baseline
the new scorer has to beat, because "semantic matching is better" is worthless
without a number attached.

**Engine** — structured LLM resume extraction; JD → machine-readable
requirements; retrieve-then-verify skill matching; GitHub analysis; LLM rubric
on redacted text; explainable weighted ensemble; confidence separate from score;
evaluation harness; fairness tooling.

**Correctness bugs fixed**

| Bug | Consequence |
|---|---|
| Campaign state in JSON files on ephemeral disk | All candidate data lost on every redeploy |
| `save_state()` read-modify-write, no locking | Concurrent gunicorn workers lost each other's writes |
| ~15 bare `except: pass` | Failed downloads scored 0, indistinguishable from a weak candidate |
| Candidates without a parseable resume link | Silently vanished from the dashboard entirely |
| Naive `datetime` → `.astimezone()` | Interview invites shifted by the server's timezone |
| Score `/5` in the UI, `/9` in the code | "7/5" displayed for strong candidates |
| Long sync inside the request cycle | 502 at gunicorn's 120s timeout on any real campaign |
| `send_outcomes` emailed *every* candidate | Second click re-mailed everyone, including the already-rejected |
| `/sync-responses/` was a GET link | Expensive irreversible action, prefetchable and CSRF-able |
| OAuth refresh tokens in plaintext | A DB dump was full Gmail/Drive access for every user |
| `DEBUG` defaulted to `True` | Debug console exposed if the env var was unset |
| `ALLOWED_HOSTS` contained `'*'` | Host-header validation disabled |
| `SECRET_KEY` fallback committed in source | Predictable session signing key |
| Broken `.gitignore` (lost `#` markers) | `pycache/` never matched `__pycache__/` |
| 46 MB of vendored wheels committed | Including Windows and linux-armv7l binaries |

---

## Honest limitations

1. **n=14, synthetic, and underpowered.** The evaluation demonstrates the
   pipeline and the trap behaviour, but the paired bootstrap shows the gap over
   the keyword baseline is *not* significant at 95% confidence — roughly 60–100
   labelled candidates would be needed. It does not establish that the weights
   generalise. Treat the scorer as a ranking aid, never an automated filter.
2. **Local inference is slow** — roughly 25–30 s per candidate on `qwen3:14b`
   (extraction + verification + rubric). Hosted models or a smaller local model
   trade accuracy for speed; `RUBRIC_ENABLED=False` is ~3× faster and measurably
   worse.
3. **Background jobs are an in-process thread pool, not Celery.** No broker
   needed, but jobs do not survive a restart (they are reaped and marked failed)
   and there is no retry. `jobs.py` is written so swapping in Celery touches
   only that file. At real scale, Celery is the right answer.
4. **The LLM judge is one weighted input, not an oracle.** A single model call
   is too unstable to hand a hiring decision to. Every rubric dimension must
   cite a verbatim quote, which is the cheapest available check on
   confabulation — but it is a check, not a guarantee.
5. **Redaction reduces the direct identity cue, not every proxy.** A resume can
   still leak identity through a location, a language, an organisation. The
   adverse-impact audit exists precisely because redaction alone is not proof of
   fairness.
6. **LinkedIn posting uses a manually-pasted token**, not an OAuth flow, which
   would require a reviewed LinkedIn application.
7. **Rotating `SECRET_KEY` invalidates stored Google tokens** unless
   `FIELD_ENCRYPTION_KEY` is set explicitly — every user would have to reconnect.

---

## Talking points

- **The measurement that changed the design.** Cosine similarity looked
  obviously right for skill matching. Measuring it showed the classes overlap
  (`python↔java` = 0.680 beats `docker↔containerisation` = 0.546), best possible
  threshold F1 = 0.700. That is what motivated retrieve-then-verify — the
  architecture came from data, not from fashion.
- **Absence vs evidence.** Splitting *score* from *confidence*, and dropping
  unavailable dimensions instead of zeroing them, is the difference between "we
  assessed this person and they are weak" and "we could not read their resume".
  The old system collapsed both into a zero.
- **Keeping the old scorer as a baseline.** Deleting it would have made the
  improvement unmeasurable.
- **Reporting a negative result about my own work.** The scorer wins on every
  metric, and I still report that the difference is not significant at this
  sample size, with the number of labels it would take to settle it. A point
  estimate that cannot survive a paired bootstrap is not a result yet.
- **Knowing when *not* to use the model.** Emails and GitHub handles are
  extracted with regex — exact, free, instant. A hallucinated GitHub username
  means scoring the wrong person's code, which is the worst failure a hiring
  tool can have.
- **Where the linear model breaks.** Additive scoring treated a missing core
  competence as outvotable. Recognising that hard requirements are conjunctive,
  and fixing it with a smooth ramp rather than a hard filter, is a modelling
  judgement rather than a coding one.
- **Building the feedback loop before it is needed.** Recruiter ratings are
  collected in the normal review flow so a real evaluation set accumulates
  without anyone doing a labelling project.

---

## Licence & data handling

Candidate resumes are personal data. They are stored per-owner per-campaign
under `media/resumes/user_{id}/campaign_{id}/`, are git-ignored, and are deleted
with the campaign. Do not commit `media/`, `.env`, `client_secrets.json` or any
real applicant data.
