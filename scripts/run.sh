pkill gunicorn
git pull
source /root/owl-server/bin/activate
gunicorn main:app --bind 0.0.0.0:443 --certfile /etc/letsencrypt/live/ninoxstrenua.site/fullchain.pem --keyfile /etc/letsencrypt/live/ninoxstrenua.site/privkey.pem --daemon

# Certificate is saved at: /etc/letsencrypt/live/ninoxstrenua.site/fullchain.pem
# Key is saved at:         /etc/letsencrypt/live/ninoxstrenua.site/privkey.pem