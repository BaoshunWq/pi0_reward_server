#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Union
import io
import math
import os
import random
import warnings
import numpy as np
import cv2
from PIL import Image as PILImage
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor


FORS_IMAGE_PATH = ["/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost1.png",
                                             "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost2.png",
                                             "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost3.png",
                                             "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost4.jpg",
                                             "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost5.jpg",
                                             "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/MM_Robustness/image_perturbation/frost6.jpg"]


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)



def _to_numpy_rgb(img: Union[np.ndarray, PILImage.Image]) -> np.ndarray:
    """Accept numpy (RGB/BGR) or PIL, return uint8 RGB ndarray."""
    if isinstance(img, PILImage.Image):
        arr = np.array(img.convert("RGB"))
        return arr
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 3:
        # 猜测是否是 BGR（OpenCV 读进来一般是 BGR）
        # 简易启发：若均值在蓝通道显著更高/更低可能为 BGR——这里直接统一按 BGR->RGB 处理更稳
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = arr.astype(np.uint8)
    return arr


def _ensure_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0, 255)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr


def _disk(radius: int, alias_blur: float = 0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 9)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased /= np.sum(aliased)
    return cv2.GaussianBlur(aliased, ksize=ksize, sigmaX=alias_blur)


def _plasma_fractal(mapsize=1024, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0), "mapsize must be power of two"
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        corner = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        acc = corner + np.roll(corner, -1, axis=0)
        acc += np.roll(acc, -1, axis=1)
        maparray[stepsize//2:mapsize:stepsize, stepsize//2:mapsize:stepsize] = wibbledmean(acc)

    def filldiamonds():
        ms = maparray.shape[0]
        dr = maparray[stepsize//2:ms:stepsize, stepsize//2:ms:stepsize]
        ul = maparray[0:ms:stepsize, 0:ms:stepsize]
        ltsum = dr + np.roll(dr, 1, axis=0) + ul + np.roll(ul, -1, axis=1)
        maparray[0:ms:stepsize, stepsize//2:ms:stepsize] = wibbledmean(ltsum)
        ttsum = dr + np.roll(dr, 1, axis=1) + ul + np.roll(ul, -1, axis=0)
        maparray[stepsize//2:ms:stepsize, 0:ms:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def _clipped_zoom(img, zoom_factor):
    if scizoom is None:
        raise ImportError("scipy.ndimage.zoom 未安装，zoom_blur 等需要它。")
    h, w = img.shape[:2]
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))
    top1 = (h - ch) // 2
    top2 = (w - cw) // 2
    out = scizoom(img[top1:top1 + ch, top2:top2 + cw], (zoom_factor, zoom_factor, 1), order=1)
    trim1 = (out.shape[0] - h) // 2
    trim2 = (out.shape[1] - w) // 2
    return out[trim1:trim1 + h, trim2:trim2 + w]


def _motion_blur_cv(img, degree, angle):
    """OpenCV 版本的线性运动模糊（无 wand 时的替代）"""
    k = np.zeros((degree, degree), dtype=np.float32)
    k[(degree - 1)//2, :] = 1.0
    # 旋转核
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    k = cv2.warpAffine(k, M, (degree, degree))
    k /= np.sum(k) if np.sum(k) != 0 else 1.0
    return cv2.filter2D(img, -1, k)


def _jpeg_compress(img_rgb, quality):
    pil = PILImage.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))


class ImagePerturber:
    """
    通用图像扰动器（不做 I/O）：
      methods: 'gaussian_noise','shot_noise','impulse_noise','speckle_noise',
               'gaussian_blur','defocus_blur','glass_blur','zoom_blur',
               'motion_blur','fog','frost','snow','spatter',
               'contrast','brightness','saturate','pixelate',
               'elastic_transform','jpeg_compression'
    """

    def __init__(
        self,
        method: str,
        severity: int = 3,
        frost_assets: Optional[List[str]] = None,  # ['frost1.png', ...]
        seed: int = 1234
    ):
        self.method = method
        self.severity = int(np.clip(severity, 1, 5))
        self.frost_assets = frost_assets 
        random.seed(seed)
        np.random.seed(seed)

        # 参数表（与你原脚本保持一致）
        self._c_gauss_noise  = [.08, .12, 0.18, 0.26, 0.38]
        self._c_shot_noise   = [60, 25, 12, 5, 3]
        self._c_impulse      = [.03, .06, .09, 0.17, 0.27]
        self._c_speckle      = [.15, .2, 0.35, 0.45, 0.6]
        self._c_gauss_blur   = [1, 2, 3, 4, 6]
        self._c_defocus      = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        self._c_zoom         = [np.arange(1, 1.11, 0.01),
                                np.arange(1, 1.16, 0.01),
                                np.arange(1, 1.21, 0.02),
                                np.arange(1, 1.26, 0.02),
                                np.arange(1, 1.33, 0.03)]
        self._c_fog          = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)]
        self._c_snow         = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
                                (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
                                (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
                                (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
                                (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)]
        self._c_spatter      = [(0.65, 0.3, 4, 0.69, 0.6, 0),
                                (0.65, 0.3, 3, 0.68, 0.6, 0),
                                (0.65, 0.3, 2, 0.68, 0.5, 0),
                                (0.65, 0.3, 1, 0.65, 1.5, 1),
                                (0.67, 0.4, 1, 0.65, 1.5, 1)]
        self._c_contrast     = [0.4, .3, .2, .1, .05]
        self._c_brightness   = [.1, .2, .3, .4, .5]
        self._c_saturate     = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)]
        self._c_pixelate     = [0.6, 0.5, 0.4, 0.3, 0.25]
        self._c_elastic      = [(244*2, 244*0.7, 244*0.1),
                                (244*2, 244*0.08, 244*0.2),
                                (244*0.05, 244*0.01, 244*0.02),
                                (244*0.07, 244*0.01, 244*0.02),
                                (244*0.12, 244*0.01, 244*0.02)]
        self._c_glass        = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)]
        self._c_motion       = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]  # (radius,length≈sigma)

    # =================== Public API ===================
    def apply(self, image: Union[np.ndarray, PILImage.Image]) -> np.ndarray:
        img = _to_numpy_rgb(image).astype(np.float32)
        method = self.method.lower()
        if   method == 'gaussian_noise':    out = self._gaussian_noise(img)
        elif method == 'shot_noise':        out = self._shot_noise(img)
        elif method == 'impulse_noise':     out = self._impulse_noise(img)
        elif method == 'speckle_noise':     out = self._speckle_noise(img)
        elif method == 'gaussian_blur':     out = self._gaussian_blur(img)
        elif method == 'defocus_blur':      out = self._defocus_blur(img)
        elif method == 'zoom_blur':         out = self._zoom_blur(img)
        elif method == 'fog':               out = self._fog(img)
        elif method == 'glass_blur':        out = self._glass_blur(img)
        elif method == 'motion_blur':       out = self._motion_blur(img)
        elif method == 'jpeg_compression':  out = self._jpeg_compression(img)
        elif method == 'frost':             out = self._frost(img)
        elif method == 'snow':              out = self._snow(img)
        elif method == 'spatter':           out = self._spatter(img)
        elif method == 'contrast':          out = self._contrast(img)
        elif method == 'brightness':        out = self._brightness(img)
        elif method == 'saturate':          out = self._saturate(img)
        elif method == 'pixelate':          out = self._pixelate(img)
        elif method == 'elastic_transform': out = self._elastic_transform(img)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return _ensure_uint8_rgb(out)

    # =================== Corruptions ===================
    def _gaussian_noise(self, x):
        c = self._c_gauss_noise[self.severity-1]
        x = x / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def _shot_noise(self, x):
        c = self._c_shot_noise[self.severity-1]
        x = x / 255.
        return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    def _impulse_noise(self, x):
        if sk is None:
            raise ImportError("需要 scikit-image 才能使用 impulse_noise")
        c = self._c_impulse[self.severity-1]
        x = sk.util.random_noise(x / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255

    def _speckle_noise(self, x):
        c = self._c_speckle[self.severity-1]
        x = x / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def _gaussian_blur(self, x):
        if gaussian is None:
            raise ImportError("需要 scikit-image 才能使用 gaussian_blur")
        c = self._c_gauss_blur[self.severity-1]
        x = gaussian(x / 255., sigma=c)
        return np.clip(x, 0, 1) * 255

    def _defocus_blur(self, x):
        c = self._c_defocus[self.severity-1]
        kernel = _disk(radius=c[0], alias_blur=c[1])
        out = np.stack([cv2.filter2D((x/255.)[:, :, d], -1, kernel) for d in range(3)], axis=-1)
        return np.clip(out, 0, 1) * 255

    def _zoom_blur(self, x):
        if scizoom is None:
            raise ImportError("需要 scipy.ndimage 才能使用 zoom_blur")
        c = self._c_zoom[self.severity-1]
        base = (x / 255.).astype(np.float32)
        acc = np.zeros_like(base)
        for z in c:
            acc += _clipped_zoom(base, z)
        out = (base + acc) / (len(c) + 1)
        return np.clip(out, 0, 1) * 255

    def _fog(self, x):
        h, w = x.shape[:2]
        c = self._c_fog[self.severity-1]
        base = x / 255.
        tmp = c[0] * _plasma_fractal(mapsize=1<<int(np.ceil(np.log2(max(h, w)))) , wibbledecay=c[1])[:h, :w][..., None]
        out = base + tmp
        max_val = base.max()
        return np.clip(out * max_val / (max_val + c[0] + 1e-8), 0, 1) * 255

    def _glass_blur(self, x):
        if gaussian is None:
            raise ImportError("需要 scikit-image 才能使用 glass_blur")
        h, w = x.shape[:2]
        c = self._c_glass[self.severity-1]  # (sigma, max_delta, iters)
        y = np.uint8(gaussian(x / 255., sigma=c[0]) * 255)
        for _ in range(c[2]):
            for i in range(c[1], h - c[1]):
                for j in range(c[1], w - c[1]):
                    dx, dy = np.random.randint(-c[1], c[1]+1, size=(2,))
                    ii, jj = i + dy, j + dx
                    y[i, j], y[ii, jj] = y[ii, jj], y[i, j]
        out = gaussian(y / 255., sigma=c[0])
        return np.clip(out, 0, 1) * 255
    def _motion_blur(self, x):
        length, sigma = self._c_motion[self.severity-1]
        angle = float(np.random.uniform(-45, 45))

        # 把原图送进 wand
        pil = PILImage.fromarray(_ensure_uint8_rgb(x))
        buf = io.BytesIO(); pil.save(buf, format='PNG'); buf.seek(0)
        with MotionImage(blob=buf.getvalue()) as wimg:
            wimg.motion_blur(radius=length, sigma=sigma, angle=angle)
            png = wimg.make_blob(format='PNG')

        # 解码回 numpy（RGB）
        arr = np.frombuffer(png, np.uint8)
        bl_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        bl_rgb = cv2.cvtColor(bl_bgr, cv2.COLOR_BGR2RGB)
        return bl_rgb
        # else:
        #     return _motion_blur_cv(x.astype(np.uint8), degree=max(3, int(length)), angle=angle)

    def _jpeg_compression(self, x):
        q = {1: 35, 2: 30, 3: 25, 4: 20, 5: 15}[self.severity]
        return _jpeg_compress(_ensure_uint8_rgb(x), q)

    def _frost(self, x):
        c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][self.severity-1]

        path = random.choice(self.frost_assets)
        frost = cv2.imread(path, cv2.IMREAD_COLOR)

        frost = cv2.cvtColor(frost, cv2.COLOR_BGR2RGB)
        fh, fw = frost.shape[:2]
        h, w = x.shape[:2]
        # 裁剪或缩放到与输入一致
        if fh < h or fw < w:
            scale = max(h/fh, w/fw)
            frost = cv2.resize(frost, (int(fw*scale), int(fh*scale)), interpolation=cv2.INTER_AREA)
        frost = frost[:h, :w]
        return np.clip(c[0] * x + c[1] * frost, 0, 255)

    def _snow(self, x):
        if gaussian is None:
            raise ImportError("需要 scikit-image 才能使用 snow")
        h, w = x.shape[:2]
        c = self._c_snow[self.severity-1]
        base = x.astype(np.float32) / 255.
        snow_layer = np.random.normal(size=(h, w), loc=c[0], scale=c[1])
        # 放大雪花
        if scizoom is None:
            raise ImportError("需要 scipy.ndimage 才能使用 snow")
        snow_layer = _clipped_zoom(snow_layer[..., None], c[2]).squeeze()
        snow_layer[snow_layer < c[3]] = 0
        # 模糊雪花(用 wand 更好，这里用线性 motion 核替代)
        motioned = _motion_blur_cv((snow_layer*255).astype(np.uint8), degree=int(c[4]), angle=float(np.random.uniform(-135, -45)))
        motioned = motioned.astype(np.float32)/255.
        # 降彩提亮
        gray = cv2.cvtColor((base*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
        base2 = c[6] * base + (1 - c[6]) * np.maximum(base, gray[..., None] * 1.5 + 0.5)
        out = np.clip(base2 + motioned[..., None] + np.rot90(motioned, 2)[..., None], 0, 1) * 255
        return out

    def _spatter(self, x):
        if gaussian is None:
            raise ImportError("需要 scikit-image 才能使用 spatter")
        c = self._c_spatter[self.severity-1]
        base = x.astype(np.float32) / 255.
        liquid = np.random.normal(size=base.shape[:2], loc=c[0], scale=c[1])
        liquid = gaussian(liquid, sigma=c[2])
        liquid[liquid < c[3]] = 0
        if c[5] == 0:
            liquid_u8 = (liquid * 255).astype(np.uint8)
            edges = 255 - cv2.Canny(liquid_u8, 50, 150)
            dist = cv2.distanceTransform(edges, cv2.DIST_L2, 5).astype(np.float32)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            ker = np.array([[-2, -1, 0],
                            [-1,  1, 1],
                            [ 0,  1, 2]], dtype=np.float32)
            dist = cv2.filter2D(dist, ddepth=-1, kernel=ker)  # 关键修复
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor((liquid * dist).astype(np.float32), cv2.COLOR_GRAY2RGB)
            m /= (np.max(m) + 1e-8)
            m *= c[4]

            color = np.array([175, 238, 238], dtype=np.float32)/255.0
            color = np.broadcast_to(color, m.shape)

            out = np.clip(base + m * color, 0, 1) * 255
            return out
        else:
            m = (liquid > c[3]).astype(np.float32)
            m = gaussian(m, sigma=c[4])
            m[m < 0.8] = 0
            color = np.stack([63, 42, 20], axis=0).astype(np.float32)/255.
            color = color[None, None, :] * m[..., None]
            base = base * (1 - m[..., None])
            return np.clip(base + color, 0, 1) * 255

    def _contrast(self, x):
        c = self._c_contrast[self.severity-1]
        x = x / 255.
        mean = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - mean) * c + mean, 0, 1) * 255

    def _brightness(self, x):
        if sk is None:
            raise ImportError("需要 scikit-image 才能使用 brightness")
        c = self._c_brightness[self.severity-1]
        x = x / 255.
        hsv = sk.color.rgb2hsv(x)
        hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
        return np.clip(sk.color.hsv2rgb(hsv), 0, 1) * 255

    def _saturate(self, x):
        if sk is None:
            raise ImportError("需要 scikit-image 才能使用 saturate")
        s, b = self._c_saturate[self.severity-1]
        x = x / 255.
        hsv = sk.color.rgb2hsv(x)
        hsv[..., 1] = np.clip(hsv[..., 1] * s + b, 0, 1)
        return np.clip(sk.color.hsv2rgb(hsv), 0, 1) * 255

    def _pixelate(self, x):
        h, w = x.shape[:2]
        c = self._c_pixelate[self.severity-1]
        small = cv2.resize(x, (max(1, int(w*c)), max(1, int(h*c))), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return big

    def _elastic_transform(self, x):
        if gaussian is None or map_coordinates is None:
            raise ImportError("需要 scikit-image 和 scipy.ndimage 才能使用 elastic_transform")
        (dx_mag, sig, aff) = self._c_elastic[self.severity-1]
        image = x.astype(np.float32) / 255.
        shape = image.shape
        center_square = np.array(shape[:2]) // 2
        square_size = min(shape[:2]) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-aff, aff, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        warped = cv2.warpAffine(image, M, (shape[1], shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        dx = gaussian(np.random.uniform(-1, 1, size=shape[:2]), sig, mode='reflect', truncate=3) * dx_mag
        dy = gaussian(np.random.uniform(-1, 1, size=shape[:2]), sig, mode='reflect', truncate=3) * dx_mag
        dx, dy = dx[..., None], dy[..., None]
        xgrid, ygrid, zgrid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='xy')
        indices = (np.reshape(ygrid + dy, (-1, 1)),
                   np.reshape(xgrid + dx, (-1, 1)),
                   np.reshape(zgrid, (-1, 1)))
        out = map_coordinates(warped, indices, order=1, mode='reflect').reshape(shape)
        return np.clip(out, 0, 1) * 255


# -------- 便捷函数式 API --------
def perturb_image(
    image: Union[np.ndarray, PILImage.Image],
    method: str,
    severity: int = 3,
    frost_assets: Optional[List[str]] = None,
    seed: int = 1234
) -> np.ndarray:
    """
    一次性调用：把 image 以指定 method/severity 扰动后返回（RGB uint8）。
    用法：
        out = perturb_image(img, method='gaussian_noise', severity=3)
        out = perturb_image(img, method='frost', severity=4, frost_assets=['frost1.png','frost2.png'])
    """

    

    return ImagePerturber(method=method, severity=severity, frost_assets=FORS_IMAGE_PATH, seed=seed).apply(image)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img_path = "libero-init-frames/libero_object_task-0_img-0_frame0.png"  # 替换为你的测试图片路径
    img = PILImage.open(img_path).convert("RGB")

    methods = [
        'gaussian_noise','shot_noise','impulse_noise','speckle_noise',
        'gaussian_blur','defocus_blur','glass_blur','zoom_blur',
        'motion_blur','fog','frost','snow','spatter',
        'contrast','brightness','saturate','pixelate',
        'elastic_transform','jpeg_compression'
    ]

    plt.figure(figsize=(15, 8))
    plt.subplot(3, 7, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    for i, method in enumerate(methods):
        perturbed = perturb_image(img, method=method, severity=1)
        plt.subplot(3, 7, i + 2)
        plt.imshow(perturbed)
        plt.title(method.replace('_', '\n'))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("./perturbed_examples.png", dpi=200)

