#!/usr/bin/env bash
# 强制清理所有相关进程

echo "强制停止所有服务器和负载均衡器..."

# 停止所有相关进程
pkill -9 -f "serve_policy" 2>/dev/null || true
pkill -9 -f "app_pi0_libero" 2>/dev/null || true
pkill -9 -f "waitress" 2>/dev/null || true
pkill -9 -f "load_balancer" 2>/dev/null || true

sleep 2

# 检查是否还有残留
REMAINING=$(ps aux | grep -E "serve_policy|app_pi0_libero|waitress|load_balancer" | grep -v grep || true)

if [[ -z "$REMAINING" ]]; then
    echo "✅ 所有进程已停止"
else
    echo "⚠️  仍有进程运行:"
    echo "$REMAINING"
fi
