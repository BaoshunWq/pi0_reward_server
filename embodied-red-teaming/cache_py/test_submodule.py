from pathlib import Path
import os
from PIL import Image as PILImage
import cv2

path = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost1.png"

p = Path(path)
print("cwd =", os.getcwd())
print("path repr =", repr(path))           # 看看有没有 \n 或空格
print("exists?", p.exists(), "is_file?", p.is_file())
print("abs =", str(p.resolve()))
print("readable?", os.access(str(p), os.R_OK))

# 能否用 PIL 打开（若 PIL 也报错，多半文件损坏）
try:
    with PILImage.open(str(p)) as im:
        im.verify()
    print("PIL verify: OK")
except Exception as e:
    print("PIL verify FAILED:", e)

# OpenCV 是否“识别”这个后缀（4.5+ 才有）
try:
    print("cv2.haveImageReader?", cv2.haveImageReader(str(p)))
except Exception:
    pass
