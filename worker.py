"""
PyWorker wrapper for Face Search Server
"""
import os
from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

# Your face server will run on port 18000
worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18000,
    model_log_file="/var/log/face-server.log",
    
    handlers=[
        # /index endpoint - indexing faces
        HandlerConfig(
            route="/index",
            method="POST",
            allow_parallel_requests=True,  # Can handle concurrent requests
            max_queue_time=120.0,  # Allow up to 2 minutes in queue
            workload_calculator=lambda payload: 100.0,  # Fixed cost per index request
            benchmark_config=None,  # Benchmark only on /search
        ),
        
        # /search endpoint - searching for faces
        HandlerConfig(
            route="/search",
            method="POST",
            allow_parallel_requests=True,
            max_queue_time=60.0,
            workload_calculator=lambda payload: float(payload.get("threshold", 0.45)) * 1000,  # Cost based on threshold
            benchmark_config=BenchmarkConfig(
                # Benchmark payload - adjust with real test image URL
                generator=lambda: {
                    "image_url": "https://link.storjshare.io/raw/jvzvldih7dypc4ra2ats62jmtgzq/family/chechi-engagement/standard-quality/AVC 0051.JPG",
                    "threshold": 0.5
                },
                runs=5,
                concurrency=4,
            ),
        ),
        
        # /health endpoint
        HandlerConfig(
            route="/health",
            method="GET",
            allow_parallel_requests=True,
            max_queue_time=10.0,
            workload_calculator=lambda payload: 1.0,  # Minimal cost
            benchmark_config=None,
        ),
    ],
    
    log_action_config=LogActionConfig(
        # Pattern to detect when your server is ready
        on_load=["Application startup complete"],
        # Patterns to detect errors
        on_error=["Traceback (most recent call last):", "ERROR", "CRITICAL"],
        # Info patterns (optional)
        on_info=["INFO", "Model loaded"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
