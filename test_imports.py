#!/usr/bin/env python
"""测试导入是否正常"""

print("测试导入...")

try:
    from pi0_reward_server.app_pi0_libero import create_app
    print("✅ app_pi0_libero 导入成功")
except Exception as e:
    print(f"❌ app_pi0_libero 导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from pi0_reward_server.reward_core import compute_score
    print("✅ reward_core 导入成功")
except Exception as e:
    print(f"❌ reward_core 导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from pi0_reward_server.utils import _extract_text_from_vllm
    print("✅ utils 导入成功")
except Exception as e:
    print(f"❌ utils 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n所有导入测试完成！")
