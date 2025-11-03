#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from typing import List, Optional
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer, util
from styleformer import Styleformer  # pip install git+https://github.com/PrithivirajDamodaran/Styleformer.git

# 需要你把 eda.py 放在同目录，暴露 eda(sentence, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug)
from eda import eda
import re


def _set_seed(seed: int = 1234):
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ### NEW: 直接内置 vla_edit 的逻辑
DEFAULT_PROTECT = {
    'put','place','set','pick','insert','remove','open','close','move','push','pull','pour','grasp','drop',
    'left','right','front','back','behind','in','into','on','onto','under','over','between',
    'bowl','mug','cup','plate','drawer','door','fridge','cabinet','microwave','book','lamp'
}

REL_SYNONYMS = [
    (r'\bnext to\b', ['beside','adjacent to','by']),
    (r'\bon the left of\b', ['to the left of']),
    (r'\bon the right of\b', ['to the right of']),
    (r'\bon the left\b', ['to the left','on left side']),
    (r'\bon the right\b', ['to the right','on right side']),
    (r'\bin\b', ['inside']),
    (r'\binto\b', ['in','inside']),
    (r'\bon\b', ['on top of'])
]

VERB_SYNONYMS = [
    (r'\bput\b', ['place','set']),
    (r'\bplace\b', ['put','set']),
    (r'\bpick up\b', ['grab','lift']),
    (r'\bremove\b', ['take out','extract']),
    (r'\binsert\b', ['put into','place inside'])
]

STOP_FUNCT = {'the','a','an'}

def _protect_tokens_vla(s: str):
    # 含数字/连字符的 token 一律保护（如 drawer-2）
    toks = re.findall(r'[a-z0-9\-]+', s.lower())
    return {t for t in toks if any(ch.isdigit() for ch in t) or '-' in t}

def slot_safe_paraphrase(
    s: str,
    protect=None,
    change_rel=True,
    change_verb=True,
    reorder_pp=True,
    polite_toggle=True,
    drop_articles=True,
) -> str:
    orig = s.strip()
    s = orig

    prot = set(DEFAULT_PROTECT if protect is None else protect)
    prot |= _protect_tokens_vla(s)

    # 1) 关系词替换（不触碰保护词）
    def safe_sub(s, pattern, choices):
        def repl(m):
            span = m.group(0)
            if span.lower() in prot:
                return span
            return random.choice(choices)
        return re.sub(pattern, repl, s, flags=re.I)


    if change_rel:
        for pat, chs in REL_SYNONYMS:
            s = safe_sub(s, pat, chs)

    # 2) 动词替换
    if change_verb:
        for pat, chs in VERB_SYNONYMS:
            s = safe_sub(s, pat, chs)

    # 3) 词序改写：把末尾 “PP”（如 next to the lamp / on the left）移到句首（或反之）
    if reorder_pp:
        m = re.search(
            r'\b(?:in|into|on|onto|under|over|between|next to|beside|adjacent to|to the left(?: of)?|to the right(?: of)?)\b.*$',
            s, flags=re.I
        )
        if m and random.random() < 0.5:
            pp = m.group(0).rstrip('. ')
            head = s[:m.start()].strip().rstrip('.')
            if head:
                if not head.lower().startswith(('please',)):
                    s = f'{pp}, {head[0].lower()+head[1:]}.'  # 句首小写
                else:
                    s = f'{pp}, {head}.'
    # 4) 礼貌标记切换
    if polite_toggle and random.random() < 0.5:
        if re.match(r'^\s*please\b', s, flags=re.I):
            s = re.sub(r'^\s*please\s+', '', s, flags=re.I)
        else:
            s = 'Please ' + s[0].lower() + s[1:]
# 5) 冠词微扰（不在保护词上）
    if drop_articles and random.random() < 0.5:
        def drop_art(m):
            w = m.group(0)
            return '' if w.lower() in STOP_FUNCT else w
        s = re.sub(r'\b(the|a|an)\b\s*', drop_art, s, flags=re.I)
        s = re.sub(r'\s+', ' ', s).strip()

    # 标点清理
    s = re.sub(r'\s+,', ',', s).strip()
    if not s.endswith('.'):
        s += '.'
    return s

def ensure_constraints(s: str, must_have) -> bool:
    low = s.lower()
    return all(tok.lower() in low for tok in must_have)

class TextPerturber:
    """
    统一的文本扰动器（不负责数据加载，仅对传入文本做扰动）

    参数
    ----
    method: str
        'back_trans' | 'keyboard' | 'ocr' |
        'insert' | 'substitute' | 'swap' | 'delete' |  # RandomCharAug 的 action
        'ip' |                                       # 插入标点
        'eda_sr' | 'eda_ri' | 'eda_rs' | 'eda_rd' |  # EDA 四种
        'style'                                      # Styleformer（需配合 style_value）
    rate: int
        扰动强度（建议 1..7，内部按 0.05*rate 计算）
    style_value: Optional[int]
        Styleformer 的风格：0=formal, 1=casual, 2=passive, 3=active
    sim_threshold: Optional[float]
        语义相似度下限（0~1）。为 None 或 0 时关闭相似度门控。
    max_tries: int
        当相似度不足时的最大重试次数
    device: Optional[str]
        sentence-transformers/Styleformer 设备，如 'cpu' 或 'cuda'
    seed: int
        随机种子
    """

    PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']

    def __init__(
        self,
        method: str,
        rate: int = 1,
        style_value: Optional[int] = None,
        sim_threshold: Optional[float] = 0.85,
        max_tries: int = 100,
        device: Optional[str] = None,
        seed: int = 1234,
    ):
        _set_seed(seed)
        self.method = method
        self.rate = max(0, int(rate))
        self.style_value = style_value
        self.sim_threshold = sim_threshold if (sim_threshold and sim_threshold > 0) else None
        self.max_tries = max_tries
        self.device = device

        # 懒加载用到的增强器/模型
        self._back_trans = None
        self._styleformer = None
        self._st_model = None

        if self.sim_threshold is not None :
            self._st_model = SentenceTransformer(
                'sentence-transformers/paraphrase-mpnet-base-v2',
                device=self.device or None
            )

    # ---------- 公共 API ----------
    def perturb(self, text: str) -> str:
        """对单条文本做扰动"""
        if not text:
            return text
        if self.sim_threshold is None:
            return self._augment_once(text)

        # 带相似度门控的重试
        base_emb = self._encode(text)
        out = text
        for _ in range(self.max_tries):
            out = self._augment_once(text)
            if self._similar_enough(base_emb, out):
                return out
        return out  # 到达最大次数也返回最后一次结果

    def perturb_many(self, texts: List[str]) -> List[str]:
        """对一组文本做扰动（保持一一对应）"""
        return [self.perturb(t) for t in texts]

    # # ---------- COCO 风格的小适配器（可选） ----------
    # def perturb_coco_annotations(self, annotations: List[dict], captions_key: str = 'caption', k_first: int = 5):
    #     """
    #     直接处理形如 COCO 的 annotation 结构：每条 ann 里有 captions list
    #     """
    #     for ann in annotations:
    #         caps = ann.get(captions_key, [])
    #         n = min(k_first, len(caps)) if k_first else len(caps)
    #         for i in range(n):
    #             caps[i] = self.perturb(caps[i])
    #     return annotations

    # ---------- 内部实现 ----------
    def _augment_once(self, s: str) -> str:
        ratio = 0.05 * self.rate

        if self.method == 'back_trans':
            if self._back_trans is None:
                self._back_trans = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de',
                    to_model_name='facebook/wmt19-de-en'
                )
            out = self._back_trans.augment(s)
            return out if isinstance(out, str) else (out[0] if out else s)
        
        # char-level augmenters
        if self.method == 'keyboard':
            aug = nac.KeyboardAug(aug_word_p=ratio)
            out = aug.augment(s)
            return out if isinstance(out, str) else (out[0] if out else s)

        if self.method == 'ocr':
            aug = nac.OcrAug(aug_word_p=ratio)
            out = aug.augment(s)
            return out if isinstance(out, str) else (out[0] if out else s)

        if self.method in {'insert', 'substitute', 'swap', 'delete'}:
            aug = nac.RandomCharAug(action=self.method, aug_word_p=ratio)
            out = aug.augment(s)
            return out if isinstance(out, str) else (out[0] if out else s)

        if self.method == 'ip':  # 插入标点
            words = s.split(' ')
            if not words:
                return s
            q = max(1, int(ratio * len(words)))
            idxs = set(random.sample(range(len(words)), min(q, len(words))))
            out = []
            for i, w in enumerate(words):
                if i in idxs:
                    out.append(random.choice(self.PUNCTUATIONS))
                out.append(w)
            return ' '.join(out)
        
        # eda word-level augmenters

        if self.method in {'eda_sr', 'eda_ri', 'eda_rs', 'eda_rd'}:
            sr = ratio if self.method == 'eda_sr' else 0.0
            ri = ratio if self.method == 'eda_ri' else 0.0
            rs = ratio if self.method == 'eda_rs' else 0.0
            rd = ratio if self.method == 'eda_rd' else 0.0
            outs = eda(s, sr, ri, rs, rd, num_aug=1)
            return outs[0] if outs else s
        
        # style transfer

        if self.method == 'style':
            if self.style_value is None:
                # 默认 formal
                self.style_value = 0
            if self._styleformer is None:
                self._styleformer = Styleformer(style=self.style_value)
            out = self._styleformer.transfer(s)
            return out 
        
        if self.method == 'slot':
            return slot_safe_paraphrase(s,)

    def _encode(self, s: str):
        if self._st_model is None:
            return None
        return self._st_model.encode(s)

    def _similar_enough(self, base_emb, t: str) -> bool:
        if self._st_model is None:
            return True  # 没有相似度模型则不做门控
        aug_emb = self._st_model.encode(t)
        score = float(util.cos_sim(aug_emb, base_emb))
        return score >= float(self.sim_threshold)


# ----------- 额外：一个简洁的函数式封装（一次性用） -----------
def perturb_text(
    text: str,
    method: str,
    rate: int = 1,
    style_value: Optional[int] = None,
    sim_threshold: Optional[float] = 0.85,
    max_tries: int = 100,
    device: Optional[str] = None,
    seed: int = 1234,
) -> str:
    """
    便捷函数：内部临时创建 TextPerturber 后对单条文本做扰动
    """
    p = TextPerturber(
        method=method, rate=rate, style_value=style_value,
        sim_threshold=sim_threshold, max_tries=max_tries,
        device=device, seed=seed
    )
    return p.perturb(text)



if  __name__ == "__main__":
    # 简单测试
    texts = [
        "Put the apple in drawer-2 on the left.",
        "Move the cup to the top shelf of the cabinet.",
        "Place the book on the table next to the lamp."
    ]

    perturber = TextPerturber(method='slot', rate=3, sim_threshold=0.8, style_value=3,max_tries=10, seed=42)
    for t in texts:
        print("原文: ", t)
        print("扰动: ", perturber.perturb(t))
        print()

    # 一次性函数式调用
    print(perturb_text("Take the keys from the drawer and put them on the counter.", method='keyboard', rate=2, sim_threshold=0.9))
