# 依赖
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch, re

class BiNLIScorer:
    """
    语义等价打分：min(p(entail x->y), p(entail y->x))，并用 max(contradiction) 做衰减。
    返回 [0,1]，越大越等价。
    """
    def __init__(self, model_id="MoritzLaurer/DeBERTa-v3-base-mnli", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        self.mdl.eval()
        cfg = self.mdl.config
        self.ent = cfg.label2id.get("entailment", 2)
        self.con = cfg.label2id.get("contradiction", 0)

    @torch.no_grad()
    def score(self, a_list, b_list) -> torch.Tensor:
        enc1 = self.tok(a_list, b_list, return_tensors="pt", padding=True, truncation=True).to(self.mdl.device)
        p1 = self.mdl(**enc1).logits.softmax(-1)
        enc2 = self.tok(b_list, a_list, return_tensors="pt", padding=True, truncation=True).to(self.mdl.device)
        p2 = self.mdl(**enc2).logits.softmax(-1)
        entail = torch.minimum(p1[:, self.ent], p2[:, self.ent])
        contra = torch.maximum(p1[:, self.con], p2[:, self.con])
        s = entail * torch.exp(-5.0 * contra)   # 矛盾强→打压
        return s.clamp(0,1)
