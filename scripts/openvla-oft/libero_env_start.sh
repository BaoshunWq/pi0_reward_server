#!/usr/bin/env bash
# LIBERO 环境评估服务器启动脚本
# 支持通过命令行参数控制执行参数

# 默认值
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
IS_SAVE_VIDEO="${IS_SAVE_VIDEO:-true}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-23451}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_suite_name|-t)
            TASK_SUITE_NAME="$2"
            shift 2
            ;;
        --num_trials_per_task|-n)
            NUM_TRIALS_PER_TASK="$2"
            shift 2
            ;;
        --is_save_video|-v)
            IS_SAVE_VIDEO="$2"
            shift 2
            ;;
        --policy_server_port|-p)
            POLICY_SERVER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --task_suite_name NAME      Task suite name (default: libero_goal)"
            echo "  -n, --num_trials_per_task NUM   Number of trials per task (default: 50)"
            echo "  -v, --is_save_video BOOL        Save video (true/false, default: false)"
            echo "  -p, --policy_server_port PORT   Policy server port (default: 23451)"
            echo "  -h, --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --task_suite_name libero_spatial --num_trials_per_task 100"
            echo "  $0 -t libero_goal -n 20 -v true -p 23452"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 执行 Python 脚本
python openvla-oft/experiments/robot/libero/run_libero_eval_server.py \
  --policy_server_host localhost \
  --policy_server_port "${POLICY_SERVER_PORT}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --is_save_video "${IS_SAVE_VIDEO}"
