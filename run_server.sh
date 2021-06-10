python -u server.py 4004 >> log/server-4004.log 2>&1 &

gunicorn --log-level=warning server:app -c gunicorn.conf >> log/gun.log 2>&1 &