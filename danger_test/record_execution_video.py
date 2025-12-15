#!/usr/bin/env python3
"""
录制环境执行视频
"""

import os
os.environ["LIBERO_CONFIG_PATH"] = "/root/autodl-tmp/code/attackVLA/pi0_reward_server/vla_simulator_env/liberoDanger/libero/configs"
import sys
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, "/root/autodl-tmp/code/attackVLA/pi0_reward_server/vla_simulator_env/liberoDanger")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def record_execution_video(task_suite_name="libero_10", 
                           task_id=0, 
                           num_steps=200,
                           fps=20,
                           output_dir="execution_videos",
                           camera_name="galleryview"):
    """
    录制环境执行视频（俯瞰视角）
    
    Args:
        task_suite_name: 任务套件名称
        task_id: 任务ID
        num_steps: 执行步数
        fps: 视频帧率
        output_dir: 输出目录
        camera_name: 摄像头名称 ('paperview'=俯瞰, 'galleryview'=高处斜视, 'agentview'=正面, 'frontview'=前视)
    """
    print("="*60)
    print(f"录制视频: {task_suite_name} - Task {task_id}")
    print("="*60)
    
    # 获取任务
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    
    print(f"\n任务名称: {task.name}")
    
    # 构建BDDL文件路径
    libero_root = os.path.join(os.path.dirname(__file__), 
                               '../vla_simulator_env/liberoDanger/libero/libero')
    libero_root = os.path.abspath(libero_root)  # 转换为绝对路径
    bddl_path = os.path.join(libero_root, 'bddl_files', task.problem_folder, task.bddl_file)
    
    if not os.path.exists(bddl_path):
        print(f"❌ BDDL文件不存在: {bddl_path}")
        return
    
    # 创建环境
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 480,  # 更高分辨率
        "camera_widths": 640,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "camera_names": [camera_name],
    }
    
    env = OffScreenRenderEnv(**env_args)
    
    # 检查危险区域
    danger_zones = env.env.danger_zone_manager.danger_zones
    print(f"\n检测到 {len(danger_zones)} 个危险区域:")
    for zone in danger_zones:
        bounds = zone.bounds
        print(f"  - {zone.name}: X=[{bounds['x_min']:.3f}, {bounds['x_max']:.3f}], "
              f"Y=[{bounds['y_min']:.3f}, {bounds['y_max']:.3f}]")
    
    if len(danger_zones) == 0:
        print("⚠️  警告：未检测到危险区域！")
        print(f"BDDL文件: {bddl_path}")
    
    # 重置环境
    obs = env.reset()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置视频写入器
    video_filename = os.path.join(output_dir, 
                                  f"{task_suite_name}_task{task_id}_{camera_name}.mp4")
    
    # 获取第一帧来确定视频尺寸
    first_frame = obs[f"{camera_name}_image"]
    if len(first_frame.shape) == 2:  # 灰度图
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
    height, width = first_frame.shape[:2]
    
    # 创建视频写入器
    # 先使用临时文件，稍后用ffmpeg转换
    temp_video_filename = video_filename.replace('.mp4', '_temp.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 使用MJPG编码，最兼容
    video_writer = cv2.VideoWriter(temp_video_filename, fourcc, fps, (width, height))
    
    # 检查视频写入器是否成功创建
    if not video_writer.isOpened():
        print("❌ 视频写入器创建失败！")
        env.close()
        return
    
    print(f"\n开始录制 {num_steps} 步...")
    print(f"视频尺寸: {width}x{height}, FPS: {fps}")
    print(f"摄像头: {camera_name}")
    
    # 执行并录制
    frames = []
    collision_steps = []
    
    for step in tqdm(range(num_steps), desc="执行步骤"):
        # 获取当前帧
        frame = obs[f"{camera_name}_image"]
        
        # 转换为BGR格式（OpenCV使用BGR）
        if len(frame.shape) == 2:  # 灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 添加信息叠加
        info_text = f"Step: {step}/{num_steps}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 如果在危险区域，添加警告
        if step > 0:  # 第一步没有info
            robot_state = env.env._arm_extractor.extract_full_arm_state(env.sim)
            in_danger, zone_name = env.env.danger_zone_manager.check_collision_full_arm(
                robot_state, step
            )
            
            if in_danger:
                collision_steps.append(step)
                warning_text = f"COLLISION! Zone: {zone_name}"
                cv2.putText(frame, warning_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 添加红色边框
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
        
        # 写入视频
        video_writer.write(frame)
        
        # 执行动作（随机动作）
        # 生成7维随机动作：[dx, dy, dz, droll, dpitch, dyaw, gripper]
        # 范围通常是[-1, 1]
        action = np.random.uniform(-1, 1, 7)
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\n任务完成于步骤 {step}")
            break
    
    # 释放资源
    video_writer.release()
    env.close()
    
    # 使用ffmpeg转换为MP4格式
    print("\n正在转换视频格式...")
    import subprocess
    try:
        # 使用ffmpeg转换为H264编码的MP4
        cmd = [
            'ffmpeg', '-y', '-i', temp_video_filename,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p', video_filename
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # 删除临时文件
            os.remove(temp_video_filename)
            print("✅ 视频转换完成")
        else:
            print("⚠️  ffmpeg转换失败")
            print(result.stderr)
            # 保留临时文件
            print(f"临时文件保存在: {temp_video_filename}")
    except FileNotFoundError:
        print("❌ 未安装ffmpeg！")
        print(f"临时AVI文件保存在: {temp_video_filename}")
        print("请手动转换: ffmpeg -i {temp_video_filename} -c:v libx264 -pix_fmt yuv420p {video_filename}")
    except Exception as e:
        print(f"❌ 视频转换失败: {e}")
        print(f"临时文件保存在: {temp_video_filename}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("录制完成！")
    print("="*60)
    print(f"视频文件: {video_filename}")
    print(f"总帧数: {min(step + 1, num_steps)}")
    print(f"碰撞步数: {len(collision_steps)}")
    if collision_steps:
        print(f"碰撞发生在步骤: {collision_steps[:10]}{'...' if len(collision_steps) > 10 else ''}")
    print("="*60)


def record_multiple_cameras(task_suite_name="libero_spatial", 
                            task_id=0, 
                            num_steps=200,
                            fps=20):
    """
    从多个摄像头角度录制视频（包含俯瞰视角）
    """
    cameras = ["paperview", "galleryview", "agentview"]  # 俯瞰优先
    
    for camera in cameras:
        print(f"\n录制摄像头: {camera}")
        try:
            record_execution_video(
                task_suite_name=task_suite_name,
                task_id=task_id,
                num_steps=num_steps,
                fps=fps,
                camera_name=camera
            )
        except Exception as e:
            print(f"❌ 摄像头 {camera} 录制失败: {e}")


def save_initial_images_all_suites(output_dir="initial_images", camera_name="paperview"):
    """
    保存所有任务套件的初始图像（俯瞰视角）
    
    Args:
        output_dir: 输出目录
        camera_name: 摄像头名称（默认paperview=俯瞰）
    """
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    # suites = [ 'libero_10']
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    total_images = 0
    
    for suite_name in suites:
        print(f"\n{'='*70}")
        print(f"处理套件: {suite_name}")
        print(f"{'='*70}")
        
        try:
            task_suite = benchmark_dict[suite_name]()
            suite_dir = os.path.join(output_dir, suite_name)
            os.makedirs(suite_dir, exist_ok=True)
            
            for task_id in range(task_suite.n_tasks):
                try:
                    task = task_suite.get_task(task_id)
                    print(f"  [{task_id}/{task_suite.n_tasks}] {task.name[:60]}...", end="")
                    
                    # 构建BDDL文件路径
                    libero_root = os.path.join(os.path.dirname(__file__), 
                                               '../vla_simulator_env/liberoDanger/libero/libero')
                    libero_root = os.path.abspath(libero_root)  # 转换为绝对路径
                    bddl_path = os.path.join(libero_root, 'bddl_files', task.problem_folder, task.bddl_file)
                    
                    if not os.path.exists(bddl_path):
                        print(f" ❌ BDDL文件不存在")
                        continue
                    
                    # 创建环境
                    env_args = {
                        "bddl_file_name": bddl_path,
                        "camera_heights": 480,
                        "camera_widths": 640,
                        "has_renderer": False,
                        "has_offscreen_renderer": True,
                        "camera_names": [camera_name],
                    }
                    
                    env = OffScreenRenderEnv(**env_args)
                    
                    # 重置环境获取初始状态
                    obs = env.reset()
                    
                    # 获取图像 (键名格式为 "{camera_name}_image")
                    image_key = f"{camera_name}_image"
                    if image_key in obs:
                        image = obs[image_key]
                        # 转换为BGR格式（OpenCV使用BGR）
                        if len(image.shape) == 2:  # 灰度图
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        elif image.shape[2] == 3:  # RGB
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        print(f" ❌ 图像键 {image_key} 不存在，可用键: {list(obs.keys())}")
                        env.close()
                        continue
                    
                    # 保存图像
                    image_filename = os.path.join(suite_dir, f"task{task_id:02d}_{camera_name}.png")
                    cv2.imwrite(image_filename, image)
                    
                    env.close()
                    total_images += 1
                    print(f" ✓")
                    
                except Exception as e:
                    print(f" ❌ 失败: {e}")
                    
        except Exception as e:
            print(f"❌ 套件 {suite_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"完成！共保存 {total_images} 张初始图像到 {output_dir}/")
    print(f"{'='*70}")


def record_all_tasks_quick(task_suite_name="libero_spatial", num_steps=100):
    """
    快速录制一个套件的所有任务（较少步数，俯瞰视角）
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    print(f"\n录制套件: {task_suite_name}")
    print(f"任务数量: {task_suite.n_tasks}")
    print(f"每个任务步数: {num_steps}")
    print("="*60)
    
    for task_id in range(task_suite.n_tasks):
        try:
            record_execution_video(
                task_suite_name=task_suite_name,
                task_id=task_id,
                num_steps=num_steps,
                fps=20,
                camera_name="paperview"  # 俯瞰视角
            )
        except Exception as e:
            print(f"❌ Task {task_id} 录制失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='录制环境执行视频或保存初始图像')
    parser.add_argument('--suite', default='libero_10', 
                       choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                       help='任务套件名称')
    parser.add_argument('--task_id', type=int, default=5, help='任务ID')
    parser.add_argument('--steps', type=int, default=100, help='执行步数')
    parser.add_argument('--fps', type=int, default=20, help='视频帧率')
    parser.add_argument('--camera', default='agentview',
                       choices=['paperview', 'galleryview', 'agentview', 'frontview', 'robot0_eye_in_hand'],
                       help='摄像头名称（默认paperview=俯瞰视角）')
    parser.add_argument('--multi_camera', action='store_true', 
                       help='从多个摄像头录制')
    parser.add_argument('--all_tasks', action='store_true', 
                       help='录制该suite的所有任务')
    parser.add_argument('--save_initial_images', action='store_true',default=True,
                       help='保存所有套件的初始图像（不录制视频）')
    parser.add_argument('--output_dir', default=None,
                       help='输出目录（默认：视频为execution_videos，图像为initial_images）')
    
    args = parser.parse_args()
    
    if args.save_initial_images:
        # 保存所有套件的初始图像
        output_dir = args.output_dir if args.output_dir else "initial_images"
        save_initial_images_all_suites(output_dir=output_dir, camera_name=args.camera)
    elif args.all_tasks:
        record_all_tasks_quick(args.suite, args.steps)
    elif args.multi_camera:
        record_multiple_cameras(args.suite, args.task_id, args.steps, args.fps)
    else:
        record_execution_video(
            task_suite_name=args.suite,
            task_id=args.task_id,
            num_steps=args.steps,
            fps=args.fps,
            camera_name=args.camera,
            output_dir=args.output_dir if args.output_dir else "execution_videos"
        )
