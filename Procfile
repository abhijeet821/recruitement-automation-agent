# Background jobs run in an in-process thread pool (see hiring_app/jobs.py), so
# each gunicorn worker owns its own pool and its own scoring concurrency. Two
# workers is a deliberate balance: enough to keep the UI responsive while a long
# screening job runs, few enough not to overwhelm a single model backend.
#
# The timeout is generous because a request may block on the LLM (JD drafting is
# synchronous by design — the recruiter is waiting for it). Screening itself is
# never in the request path.
#
# Moving to Celery would replace the thread pool with a separate worker process:
#   worker: celery -A my_hiring_project worker --concurrency 2
web: gunicorn my_hiring_project.wsgi --workers 2 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT --access-logfile -
