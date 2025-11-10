"""
配置管理模块 - 统一管理环境变量和默认配置
"""
import os
from typing import List, Tuple


class Config:
    """配置管理类"""
    
    # Reward Server配置
    PORT = int(os.environ.get("PORT", 6006))
    
    # GPU配置
    GPU_HOST = os.environ.get("GPU_HOST", "0.0.0.0")
    GPU_PORTS_STR = os.environ.get("GPU_PORTS", "8000,8001")
    
    @classmethod
    def get_gpu_ports(cls) -> List[int]:
        """获取GPU端口列表"""
        return [int(p.strip()) for p in cls.GPU_PORTS_STR.split(",")]
    
    @classmethod
    def get_hosts_ports(cls) -> List[Tuple[str, int]]:
        """获取(host, port)元组列表"""
        return [(cls.GPU_HOST, port) for port in cls.get_gpu_ports()]
    
    # Gunicorn配置
    GUNICORN_WORKERS = int(os.environ.get("GUNICORN_WORKERS", 2))
    GUNICORN_TIMEOUT = int(os.environ.get("GUNICORN_TIMEOUT", 1800))
    GUNICORN_THREADS = int(os.environ.get("GUNICORN_THREADS", 1))
    WORKER_CLASS = os.environ.get("WORKER_CLASS", "sync")
    MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", 100))
    
    # 日志配置
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
    ACCESS_LOG = os.environ.get("ACCESS_LOG", "-")
    ERROR_LOG = os.environ.get("ERROR_LOG", "-")
    
    # 评测任务默认配置
    DEFAULT_TASK_SUITE = os.environ.get("DEFAULT_TASK_SUITE", "libero_spatial")
    DEFAULT_NUM_TRIALS = int(os.environ.get("DEFAULT_NUM_TRIALS", 5))
    DEFAULT_NUM_STEPS_WAIT = int(os.environ.get("DEFAULT_NUM_STEPS_WAIT", 10))
    VIDEO_OUT_PATH = os.environ.get("VIDEO_OUT_PATH", "/root/autodl-tmp/output_big_data/libero/videos")
    SAVE_VIDEOS = os.environ.get("SAVE_VIDEOS", "false").lower() == "true"
    
    # 调试配置
    FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    VERBOSE = os.environ.get("VERBOSE", "false").lower() == "true"
    
    # AutoDL配置
    AUTODL_PUBLIC_URL = os.environ.get("AUTODL_PUBLIC_URL", "")
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 50)
        print("当前配置:")
        print("=" * 50)
        print(f"Reward Server Port: {cls.PORT}")
        print(f"GPU Host: {cls.GPU_HOST}")
        print(f"GPU Ports: {cls.get_gpu_ports()}")
        print(f"Gunicorn Workers: {cls.GUNICORN_WORKERS}")
        print(f"Gunicorn Timeout: {cls.GUNICORN_TIMEOUT}s")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Default Task Suite: {cls.DEFAULT_TASK_SUITE}")
        print(f"Default Num Trials: {cls.DEFAULT_NUM_TRIALS}")
        print(f"Save Videos: {cls.SAVE_VIDEOS}")
        print(f"Debug Mode: {cls.FLASK_DEBUG}")
        if cls.AUTODL_PUBLIC_URL:
            print(f"AutoDL Public URL: {cls.AUTODL_PUBLIC_URL}")
        print("=" * 50)
    
    @classmethod
    def validate(cls):
        """验证配置"""
        errors = []
        
        # 检查GPU端口
        gpu_ports = cls.get_gpu_ports()
        if len(gpu_ports) == 0:
            errors.append("GPU_PORTS不能为空")
        if len(gpu_ports) != len(set(gpu_ports)):
            errors.append("GPU_PORTS包含重复端口")
        
        # 检查端口范围
        if not (1024 <= cls.PORT <= 65535):
            errors.append(f"PORT {cls.PORT} 超出有效范围 (1024-65535)")
        
        for port in gpu_ports:
            if not (1024 <= port <= 65535):
                errors.append(f"GPU端口 {port} 超出有效范围 (1024-65535)")
        
        # 检查worker数量
        if cls.GUNICORN_WORKERS < 1:
            errors.append("GUNICORN_WORKERS必须至少为1")
        
        # 检查超时时间
        if cls.GUNICORN_TIMEOUT < 30:
            errors.append("GUNICORN_TIMEOUT太短，建议至少30秒")
        
        return errors


# 导出配置实例
config = Config()


if __name__ == "__main__":
    # 测试配置
    config.print_config()
    
    # 验证配置
    errors = config.validate()
    if errors:
        print("\n配置错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ 配置验证通过")

