# HireAI — Explainable Candidate Screening

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

| Scorer | Spearman ρ | Kendall τ | NDCG@5 | P@5 | MAE |
|---|---|---|---|---|---|
| **ensemble** (production) | **0.964** | **0.897** | **0.956** | 1.000 | 10.3 |
| `keyword_jd` (JD-aware substrings) | 0.942 | 0.888 | 0.716 | 1.000 | 10.2 |
| `keyword_legacy` (the original scorer) | 0.822 | 0.713 | 0.701 | 1.000 | 14.3 |
| `random` | −0.354 | −0.291 | 0.323 | 0.400 | 39.7 |

The headline is **NDCG@5: 0.956 vs 0.701**, a 36% relative improvement in the
quality of the top of the ranking — which is the part a recruiter actually
reads.

The set contains deliberate traps for literal matching. The clearest:

| Candidate | Human label | Legacy keyword score | Ensemble |
|---|---|---|---|
| Node.js backend dev | 3 / 5 | **88.9** (its joint-highest) | 72.3 |
| Senior Java/Spring engineer | 2 / 5 | 66.7 | 53.1 |
| Senior Python/Django engineer | 5 / 5 | 77.8 | 91.2 |

The legacy scorer ranked a Node.js developer above every genuine Python
candidate, because `node`, `api`, `rest`, `sql` and `docker` are all in its
hardcoded list.

**Read these numbers honestly.** n=14 on synthetic resumes is enough to
demonstrate that the pipeline works and that the traps behave as designed. It is
not enough to prove the weights are optimal. That is why the app collects
recruiter ratings in the review flow — see [Closing the loop](#closing-the-loop).

---

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
├── generation/              JD drafting + measurable quality grading
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

### Commands

| Command | Purpose |
|---|---|
| `manage.py doctor` | Check every dependency; says exactly what to fix |
| `manage.py evaluate_scorer` | Benchmark the scorer against all baselines |
| `manage.py evaluate_scorer --campaign N` | Benchmark against real recruiter ratings |
| `manage.py calibrate_similarity` | Re-measure embedding separation |
| `manage.py audit_fairness --campaign N` | Adverse-impact report |
| `pytest` | 117 tests, no model server required |

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

1. **n=14, synthetic.** The evaluation demonstrates the pipeline and the trap
   behaviour. It does not establish that the weights generalise. Treat the
   scorer as a ranking aid, never an automated filter.
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
