"""
Minimal RQ worker bootstrap.
Connects to Redis and listens on the "default" queue.
"""

import os
import logging
from redis import Redis
from rq import Worker, Queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get Redis URL from environment
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Connect to Redis
redis_conn = Redis.from_url(redis_url)

# Create a queue
queue = Queue(connection=redis_conn)

if __name__ == "__main__":
    logger.info(f"Starting RQ worker, connecting to {redis_url}")
    logger.info("Listening on queue: default")

    # Start the worker
    worker = Worker([queue], connection=redis_conn)
    worker.work()
