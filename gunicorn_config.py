"""
Gunicorn配置文件 - 用于生产环境部署
"""
import multiprocessing
import os

# 服务器绑定地址和端口
bind = f"0.0.0.0:{os.environ.get('PORT', 6006)}"

# Worker进程数量
# 建议：2-4 x $(NUM_CORES)，但对于GPU密集型任务，建议与GPU数量一致
workers = int(os.environ.get("GUNICORN_WORKERS", 2))

# Worker类型：gevent支持异步，适合I/O密集型
# sync适合CPU密集型，但我们使用线程池处理，所以用gevent更好
worker_class = os.environ.get("WORKER_CLASS", "sync")

# 每个worker的线程数（仅对gthread worker有效）
threads = int(os.environ.get("GUNICORN_THREADS", 1))

# 超时时间（秒）- 评测任务可能很长，设置为30分钟
timeout = int(os.environ.get("GUNICORN_TIMEOUT", 1800))

# 优雅重启的超时时间
graceful_timeout = 120

# 保持连接的超时时间
keepalive = 5

# 日志配置
accesslog = os.environ.get("ACCESS_LOG", "-")  # stdout
errorlog = os.environ.get("ERROR_LOG", "-")    # stderr
loglevel = os.environ.get("LOG_LEVEL", "info")

# 进程命名
proc_name = "pi0_reward_server"

# 最大请求数，达到后重启worker（防止内存泄漏）
max_requests = int(os.environ.get("MAX_REQUESTS", 100))
max_requests_jitter = 10

# 预加载应用（节省内存，但调试时可能需要禁用）
preload_app = os.environ.get("PRELOAD_APP", "false").lower() == "true"

# Worker临时目录
worker_tmp_dir = "/dev/shm"  # 使用内存文件系统，提高性能

# 限制请求体大小（例如1GB）
limit_request_line = 0  # 不限制请求行长度
limit_request_fields = 100
limit_request_field_size = 0  # 不限制字段大小


def on_starting(server):
    """服务器启动时的回调"""
    print(f"Starting Gunicorn server on {bind}")
    print(f"Workers: {workers}, Timeout: {timeout}s")


def worker_int(worker):
    """Worker被中断时的回调"""
    print(f"Worker {worker.pid} received SIGINT, shutting down gracefully...")


def worker_abort(worker):
    """Worker异常终止时的回调"""
    print(f"Worker {worker.pid} aborted, restarting...")

