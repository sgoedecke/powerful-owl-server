pkill gunicorn
gunicorn main:app --bind 0.0.0.0:443 --certfile /etc/letsencrypt/live/ninoxstrenua.site/fullchain.pem --keyfile /etc/letsencrypt/live/ninoxstrenua.site/privkey.pem