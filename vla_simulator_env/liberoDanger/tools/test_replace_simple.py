#!/usr/bin/env python3
"""
简单测试危险区域替换功能
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from replace_danger_zones_in_bddl import (
    find_danger_zones_in_bddl,
    extract_new_ranges_from_generated,
    replace_zone_ranges
)


def test_replace_logic():
    """测试替换逻辑"""
    
    print("=" * 80)
    print("测试危险区域ranges替换逻辑")
    print("=" * 80)
    
    # 模拟原始BDDL内容
    original_bddl = """
      (main_table_danger_zone_1
          (:target main_table)
          (:ranges (
              (0.2782 -0.1764 0.3582 -0.0964)
            )
          )
      )
      (main_table_danger_zone_2
          (:target main_table)
          (:ranges (
              (0.2984 0.0459 0.3784 0.1259)
            )
          )
      )
"""
    
    print("\n原始BDDL内容:")
    print("-" * 80)
    print(original_bddl)
    print("-" * 80)
    
    # 查找旧的危险区域
    old_zones = find_danger_zones_in_bddl(original_bddl)
    print(f"\n找到 {len(old_zones)} 个危险区域:")
    for zone in old_zones:
        print(f"  - {zone['name']}: ranges = ({zone['ranges']})")
    
    # 模拟生成的新危险区域定义
    new_zone_defs = [
        """(auto_danger_zone_1
    (:target main_table)
    (:ranges (
        (-0.0864 0.2297 -0.0064 0.3097)
    ))
)""",
        """(auto_danger_zone_2
    (:target main_table)
    (:ranges (
        (0.0934 0.1589 0.1734 0.2389)
    ))
)"""
    ]
    
    # 提取新的ranges
    new_ranges = extract_new_ranges_from_generated(new_zone_defs)
    print(f"\n提取的新ranges:")
    for i, r in enumerate(new_ranges):
        print(f"  {i+1}. ({r})")
    
    # 执行替换
    print("\n执行替换...")
    modified_bddl = replace_zone_ranges(original_bddl, old_zones, new_ranges)
    
    print("\n替换后的BDDL内容:")
    print("-" * 80)
    print(modified_bddl)
    print("-" * 80)
    
    # 验证结果
    print("\n验证结果:")
    if "main_table_danger_zone_1" in modified_bddl:
        print("✅ 保持了原有的危险区域名称 main_table_danger_zone_1")
    else:
        print("❌ 丢失了原有的危险区域名称")
    
    if "-0.0864 0.2297 -0.0064 0.3097" in modified_bddl:
        print("✅ 成功替换了第一个危险区域的ranges")
    else:
        print("❌ 未能替换第一个危险区域的ranges")
    
    if "0.0934 0.1589 0.1734 0.2389" in modified_bddl:
        print("✅ 成功替换了第二个危险区域的ranges")
    else:
        print("❌ 未能替换第二个危险区域的ranges")
    
    if "auto_danger_zone" not in modified_bddl:
        print("✅ 没有引入新的危险区域名称")
    else:
        print("❌ 引入了新的危险区域名称")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_replace_logic()
