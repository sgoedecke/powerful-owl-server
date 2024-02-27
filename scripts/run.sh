pkill gunicorn
git pull
gunicorn --bind 0.0.0.0:80 main:app --daemon