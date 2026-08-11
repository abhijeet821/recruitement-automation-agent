#!/usr/bin/env bash
# Deployment build step (Railway / Render / Heroku).
set -o errexit

pip install -r requirements.txt

# Postgres driver is not in requirements.txt so that local SQLite development
# needs no build toolchain. Install it only when a database URL is present.
if [ -n "$DATABASE_URL" ]; then
  pip install "psycopg[binary]==3.2.12"
fi

python manage.py collectstatic --no-input
python manage.py migrate --no-input

# Surface a misconfigured scoring backend at deploy time rather than on the
# first candidate. Non-fatal: the app is still usable for everything but scoring.
python manage.py doctor || echo "WARNING: doctor reported problems — see above."
