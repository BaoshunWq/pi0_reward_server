"""
多GPU客户端连接池管理器
支持在多个GPU上部署Pi0模型，并行处理请求
"""
import logging
import threading
import queue
from typing import List, Optional, Dict, Any
from openpi_client import websocket_client_policy as _websocket_client_policy
import time


class ClientPool:
    """WebSocket客户端连接池，用于管理多个GPU上的Pi0模型客户端"""
    
    def __init__(self, hosts_ports: List[tuple], max_retries: int = 3):
        """
        初始化客户端连接池
        
        Args:
            hosts_ports: [(host1, port1), (host2, port2), ...] 多个模型服务的地址
            max_retries: 连接失败时的最大重试次数
        """
        self.hosts_ports = hosts_ports
        self.max_retries = max_retries
        self.pool = queue.Queue()
        self.lock = threading.Lock()
        self.total_clients = len(hosts_ports)
        
        # 初始化所有客户端
        self._initialize_clients()
        
        logging.info(f"ClientPool initialized with {self.total_clients} clients")
    
    def _initialize_clients(self):
        """初始化所有客户端连接"""
        for idx, (host, port) in enumerate(self.hosts_ports):
            try:
                client = _websocket_client_policy.WebsocketClientPolicy(host, port)
                client_info = {
                    'client': client,
                    'host': host,
                    'port': port,
                    'idx': idx
                }
                self.pool.put(client_info)
                logging.info(f"Client {idx} initialized: {host}:{port}")
            except Exception as e:
                logging.error(f"Failed to initialize client {idx} ({host}:{port}): {e}")
    
    def acquire(self, timeout: float = 300.0) -> Optional[Dict[str, Any]]:
        """
        从池中获取一个可用的客户端
        
        Args:
            timeout: 等待超时时间（秒）
            
        Returns:
            client_info dict 或 None（如果超时）
        """
        try:
            client_info = self.pool.get(timeout=timeout)
            logging.debug(f"Client {client_info['idx']} acquired from pool")
            return client_info
        except queue.Empty:
            logging.error(f"Failed to acquire client within {timeout}s")
            return None
    
    def release(self, client_info: Dict[str, Any]):
        """
        将客户端归还到池中
        
        Args:
            client_info: 客户端信息字典
        """
        if client_info:
            self.pool.put(client_info)
            logging.debug(f"Client {client_info['idx']} released back to pool")
    
    def close_all(self):
        """关闭所有客户端连接"""
        while not self.pool.empty():
            try:
                client_info = self.pool.get_nowait()
                client = client_info['client']
                if hasattr(client, '_ws') and client._ws is not None:
                    client._ws.close()
                logging.info(f"Client {client_info['idx']} closed")
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Error closing client: {e}")


# 全局客户端池实例
_global_client_pool: Optional[ClientPool] = None
_pool_lock = threading.Lock()


def initialize_global_pool(hosts_ports: List[tuple]):
    """
    初始化全局客户端池（线程安全）
    
    Args:
        hosts_ports: [(host1, port1), (host2, port2), ...]
    """
    global _global_client_pool
    with _pool_lock:
        if _global_client_pool is None:
            _global_client_pool = ClientPool(hosts_ports)
            logging.info("Global client pool initialized")
        else:
            logging.warning("Global client pool already initialized")


def get_global_pool() -> Optional[ClientPool]:
    """获取全局客户端池实例"""
    return _global_client_pool


def cleanup_global_pool():
    """清理全局客户端池"""
    global _global_client_pool
    with _pool_lock:
        if _global_client_pool is not None:
            _global_client_pool.close_all()
            _global_client_pool = None
            logging.info("Global client pool cleaned up")

