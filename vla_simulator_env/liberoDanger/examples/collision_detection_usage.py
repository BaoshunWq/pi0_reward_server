"""
碰撞检测系统使用示例

展示如何使用改进后的碰撞检测系统，包括：
1. 自动碰撞检测（每个step自动调用）
2. 三种碰撞计数语义
3. 在奖励函数中使用碰撞信息
4. 在训练日志中记录碰撞统计
"""

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def example_basic_usage():
    """示例1: 基本使用 - 碰撞信息自动添加到info"""
    
    # 创建环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)
    
    env_args = {
        "bddl_file_name": task.problem_folder,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.reset()
    
    # 执行一些步骤
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # 碰撞信息已自动添加到info字典中
        if step % 10 == 0:
            print(f"\n步骤 {step}:")
            print(f"  进入次数: {info['collision_enter_count']}")
            print(f"  接触帧数: {info['collision_frame_count']}")
            print(f"  完整事件数: {info['collision_event_count']}")
            print(f"  当前在危险区域: {info['in_danger_zone']}")
            
            if info['last_collision_details']['collision_type']:
                print(f"  最后碰撞类型: {info['last_collision_details']['collision_type']}")
                print(f"  碰撞部位: {info['last_collision_details']['collision_body']}")
        
        if done:
            break
    
    env.close()


def example_reward_integration():
    """示例2: 在奖励函数中使用碰撞信息"""
    
    class CustomRewardEnv(OffScreenRenderEnv):
        """自定义奖励函数的环境"""
        
        def __init__(self, collision_penalty=0.1, **kwargs):
            super().__init__(**kwargs)
            self.collision_penalty = collision_penalty
            self.last_collision_enter_count = 0
        
        def _post_action(self, action):
            obs, reward, done, info = super()._post_action(action)
            
            # 根据碰撞信息调整奖励
            # 方案1: 每次进入危险区域扣分
            new_collisions = info['collision_enter_count'] - self.last_collision_enter_count
            if new_collisions > 0:
                reward -= self.collision_penalty * new_collisions
                print(f"⚠️  碰撞惩罚: -{self.collision_penalty * new_collisions}")
            
            self.last_collision_enter_count = info['collision_enter_count']
            
            # 方案2: 根据接触帧数持续扣分（更严格）
            # if info['in_danger_zone']:
            #     reward -= self.collision_penalty * 0.01  # 每帧小幅扣分
            
            # 方案3: 根据完整碰撞事件扣分
            # if info['collision_event_count'] > 0:
            #     reward -= self.collision_penalty * info['collision_event_count']
            
            return obs, reward, done, info
        
        def reset(self):
            self.last_collision_enter_count = 0
            return super().reset()
    
    # 使用自定义环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)
    
    env_args = {
        "bddl_file_name": task.problem_folder,
        "camera_heights": 128,
        "camera_widths": 128,
        "collision_penalty": 0.2,  # 每次碰撞扣0.2分
    }
    
    env = CustomRewardEnv(**env_args)
    env.reset()
    
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"\n总奖励: {total_reward}")
    print(f"总碰撞次数: {info['collision_enter_count']}")
    
    env.close()


def example_training_logging():
    """示例3: 在训练日志中记录碰撞统计"""
    
    # 创建环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)
    
    env_args = {
        "bddl_file_name": task.problem_folder,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    env = OffScreenRenderEnv(**env_args)
    
    # 模拟多个episode的训练
    num_episodes = 5
    episode_stats = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 记录episode统计信息
        stats = {
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'collision_enter_count': info['collision_enter_count'],
            'collision_frame_count': info['collision_frame_count'],
            'collision_event_count': info['collision_event_count'],
            'collision_rate': info['collision_frame_count'] / episode_length if episode_length > 0 else 0,
        }
        episode_stats.append(stats)
        
        print(f"\nEpisode {episode}:")
        print(f"  奖励: {stats['reward']:.3f}")
        print(f"  长度: {stats['length']}")
        print(f"  进入危险区域次数: {stats['collision_enter_count']}")
        print(f"  接触帧数: {stats['collision_frame_count']}")
        print(f"  完整碰撞事件数: {stats['collision_event_count']}")
        print(f"  碰撞率: {stats['collision_rate']:.2%}")
    
    # 计算平均统计
    avg_collision_enter = np.mean([s['collision_enter_count'] for s in episode_stats])
    avg_collision_frame = np.mean([s['collision_frame_count'] for s in episode_stats])
    avg_collision_event = np.mean([s['collision_event_count'] for s in episode_stats])
    avg_collision_rate = np.mean([s['collision_rate'] for s in episode_stats])
    
    print(f"\n{'='*60}")
    print(f"平均统计 ({num_episodes} episodes):")
    print(f"  平均进入次数: {avg_collision_enter:.2f}")
    print(f"  平均接触帧数: {avg_collision_frame:.2f}")
    print(f"  平均完整事件数: {avg_collision_event:.2f}")
    print(f"  平均碰撞率: {avg_collision_rate:.2%}")
    
    env.close()


def example_collision_semantics():
    """示例4: 理解三种碰撞计数语义的区别"""
    
    print("="*60)
    print("三种碰撞计数语义说明:")
    print("="*60)
    
    print("\n1. collision_enter_count (进入次数)")
    print("   - 每次机械臂进入危险区域时计数+1")
    print("   - 在危险区域内持续移动不会重复计数")
    print("   - 适用场景: 统计违规次数，每次违规扣分")
    print("   - 示例: 进入→停留10帧→离开→再进入 = 2次")
    
    print("\n2. collision_frame_count (接触帧数)")
    print("   - 每帧机械臂在危险区域内都计数+1")
    print("   - 反映在危险区域内的总时间")
    print("   - 适用场景: 持续惩罚，停留越久惩罚越大")
    print("   - 示例: 进入→停留10帧→离开→再进入停留5帧 = 15帧")
    
    print("\n3. collision_event_count (完整事件数)")
    print("   - 进入+离开算一次完整碰撞事件")
    print("   - 只有完全离开危险区域后才计数")
    print("   - 适用场景: 统计独立碰撞事件，避免重复计数")
    print("   - 示例: 进入→停留10帧→离开→再进入→离开 = 2次事件")
    
    print("\n" + "="*60)
    print("使用建议:")
    print("="*60)
    print("- 训练初期: 使用 collision_enter_count，快速反馈违规行为")
    print("- 精细控制: 使用 collision_frame_count，鼓励快速离开危险区域")
    print("- 评估指标: 使用 collision_event_count，统计独立碰撞事件数")
    print("="*60)


if __name__ == "__main__":
    print("碰撞检测系统改进示例\n")
    
    # 运行示例
    print("\n" + "="*60)
    print("示例4: 理解碰撞计数语义")
    print("="*60)
    example_collision_semantics()
    
    # 取消注释以运行其他示例
    # print("\n" + "="*60)
    # print("示例1: 基本使用")
    # print("="*60)
    # example_basic_usage()
    
    # print("\n" + "="*60)
    # print("示例2: 奖励函数集成")
    # print("="*60)
    # example_reward_integration()
    
    # print("\n" + "="*60)
    # print("示例3: 训练日志记录")
    # print("="*60)
    # example_training_logging()
