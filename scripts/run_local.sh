pkill gunicorn
git pull
gunicorn main:app -w 2 --timeout 3000 --bind 0.0.0.0:443 --certfile /etc/letsencrypt/live/ninoxstrenua.site/fullchain.pem --keyfile /etc/letsencrypt/live/ninoxstrenua.site/privkey.pem