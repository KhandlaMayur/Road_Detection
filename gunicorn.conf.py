# Gunicorn configuration file
import multiprocessing

# Worker class for handling long running tasks like YOLO inference
worker_class = 'gthread'

# 2 threads per worker is enough for our Render free instance
threads = 2
workers = 1

# Increase timeout so the worker is not killed mid-inference
timeout = 300

# Bind
bind = '0.0.0.0:10000'
