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
        # /index endpoint - indexing faces (MAIN OPERATION)
        HandlerConfig(
            route="/index",
            allow_parallel_requests=True,
            max_queue_time=120.0,
            workload_calculator=lambda payload: 100.0,  # Fixed cost per index request
            benchmark_config=BenchmarkConfig(
                # Benchmark payload for indexing - adjust with your actual index request format
                # generator=lambda: {
                #     "image_url": "https://link.storjshare.io/raw/jvzvldih7dypc4ra2ats62jmtgzq/family/chechi-engagement/standard-quality/AVC 0051.JPG",
                #     # Add any other fields your index endpoint expects:
                #     # "face_id": "benchmark_face_001",
                #     # "metadata": {"name": "Test Face"},
                # },
                dataset=[
                    {
                        "image_url": "https://link.storjshare.io/raw/jxcy5wbsh36pwrtof5e73dzs5gwq/family/chechi-engagement/standard-quality/AVC 0089.JPG",
                        "image_path": "/users/john/photo1.jpg",
                        "overwrite": "false"
                    }
                ],
                runs=5,
                concurrency=4,
            ),
        ),
        
        # /search endpoint - searching for faces
        HandlerConfig(
            route="/search",
            allow_parallel_requests=True,
            max_queue_time=60.0,
            workload_calculator=lambda payload: float(payload.get("threshold", 0.45)) * 1000,
        ),
        
        # /health endpoint
        HandlerConfig(
            route="/health",
            allow_parallel_requests=True,
            max_queue_time=10.0,
            workload_calculator=lambda payload: 1.0,
        ),
    ],
    
    log_action_config=LogActionConfig(
        on_load=["Application startup complete"],
        on_error=["Traceback (most recent call last):", "ERROR", "CRITICAL"],
        on_info=["INFO", "Model loaded"],
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
