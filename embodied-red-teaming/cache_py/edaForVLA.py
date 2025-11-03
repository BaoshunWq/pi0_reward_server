# eda_vla.py
# VLA-friendly EDA: 保留数字/连字符，保护关键词，避免删改动作/对象/方向等要害词
# 基于 https://github.com/jasonwei20/eda_nlp 做了最小入侵修改

import re
import random
from random import shuffle
random.seed(1)

# ------------------ 可按需增删的保护词（VLA常见动作、对象、方向、颜色、数量等） ------------------
PROTECTED_DEFAULT = {
    # actions
    'put','place','pick','pick-and-place','pickplace','take','insert','remove',
    'open','close','move','push','pull','pour','grasp','drop',
    # objects (常见LIBERO/OpenVLA物体名，按需补全)
    'bowl','mug','cup','plate','drawer','door','fridge','cabinet','microwave',
    'sponge','soap','bottle','banana','apple','cream','cheese','spoon','fork','knife','block',
    # directions / spatial
    'left','right','top','bottom','front','back','behind','between','middle','center',
    'in','into','onto','inside','outside','under','over','from','to','on','off',
    # colors
    'red','blue','green','yellow','orange','purple','white','black','brown','pink','gray','grey',
    # quantities
    'one','two','three','four','five','first','second','third'
}

# 原始 stop words（用于SR时不被替换），保持不变
stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which','who',
    'whom','this','that','these','those','am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or',
    'because','as','until','while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once','here','there','when',
    'where','why','how','all','any','both','each','few','more','most','other','some','such',
    'no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just',
    'don','should','now','']

# -------- 清洗函数：保留数字与连字符，避免破坏例如 "drawer-2" / "top-left" --------
def get_only_chars_vla(line, keep_digits=True, keep_hyphen=True):
    line = line.replace("’","").replace("'","").replace("\t"," ").replace("\n"," ").lower()
    # 不再把 '-' 改成空格；用正则统一保留字符集合
    allowed = 'a-z ' + ('0-9' if keep_digits else '') + ('-' if keep_hyphen else '')
    line = re.sub(f'[^{allowed}]', ' ', line)
    line = re.sub(' +', ' ', line).strip()
    return line

# ---------------- WordNet 同义词（与原版一致，仅多了过滤） ----------------
# 第一次使用需要：
# import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            s = l.name().replace("_"," ").replace("-"," ").lower()
            s = "".join([ch for ch in s if ch in ' qwertyuiopasdfghjklzxcvbnm'])  # 单词字符
            s = s.strip()
            if s and ' ' not in s:          # 仅保留单词，不要多词短语，避免"loving cup"这类
                synonyms.add(s)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def _is_protected(tok, protected_set):
    # 保护：显式词、包含数字/连字符的复合词
    if tok in protected_set:
        return True
    if any(ch.isdigit() for ch in tok):
        return True
    if '-' in tok:
        return True
    return False

# ---------------- SR / RD / RS / RI：加入保护词与更稳健的判断 ----------------
def synonym_replacement(words, n, protected_set):
    new_words = words.copy()
    candidates = [w for w in words if (w not in stop_words) and (not _is_protected(w, protected_set))]
    random.shuffle(candidates)
    num_replaced = 0
    for w in candidates:
        syns = get_synonyms(w)
        syns = [s for s in syns if (not _is_protected(s, protected_set))]
        if syns:
            s = random.choice(syns)
            new_words = [s if word == w else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_deletion(words, p, protected_set):
    if len(words) == 1:
        return words
    new_words = []
    for w in words:
        if _is_protected(w, protected_set):
            new_words.append(w)              # 保护词不删
            continue
        if random.uniform(0,1) > p:
            new_words.append(w)
    if not new_words:
        new_words = [words[random.randint(0, len(words)-1)]]
    return new_words

def swap_word(new_words, protected_set):
    if len(new_words) < 2:
        return new_words
    tries = 0
    while tries < 5:
        i = random.randint(0, len(new_words)-1)
        j = random.randint(0, len(new_words)-1)
        if i != j and (not _is_protected(new_words[i], protected_set)) and (not _is_protected(new_words[j], protected_set)):
            new_words[i], new_words[j] = new_words[j], new_words[i]
            return new_words
        tries += 1
    return new_words

def random_swap(words, n, protected_set):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words, protected_set)
    return new_words

def add_word(new_words, protected_set):
    syns = []
    cnt = 0
    while not syns:
        w = new_words[random.randint(0, len(new_words)-1)]
        if _is_protected(w, protected_set):
            cnt += 1
            if cnt >= 10:
                return
            continue
        syns = get_synonyms(w)
        syns = [s for s in syns if (not _is_protected(s, protected_set))]
        cnt += 1
        if cnt >= 10:
            return
    s = syns[0]
    idx = random.randint(0, len(new_words)-1)
    new_words.insert(idx, s)

def random_insertion(words, n, protected_set):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, protected_set)
    return new_words

# ---------------- 入口：VLA友好的 EDA ----------------
def eda_vla(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.05, num_aug=1,
            protected=None, keep_digits=True, keep_hyphen=True):
    """
    与原 eda(...) 类似，但：
      - 清洗保留数字/连字符
      - SR/RI/RS/RD 都会跳过保护词（动作/对象/方向等）与含数字/连字符的 token
      - 默认 num_aug=1 更适合在线扰动（可改）
      - RD 默认强度下调到 0.05，避免删掉关键信息
    """
    protected_set = set(PROTECTED_DEFAULT if protected is None else protected)
    sentence = get_only_chars_vla(sentence, keep_digits=keep_digits, keep_hyphen=keep_hyphen)
    words = [w for w in sentence.split(' ') if w != '']
    num_words = len(words)

    augmented = []
    num_new_per_technique = int(num_aug/4)+1

    # SR
    if alpha_sr > 0 and num_words > 0:
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a = synonym_replacement(words, n_sr, protected_set)
            augmented.append(' '.join(a))

    # RI
    if alpha_ri > 0 and num_words > 0:
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a = random_insertion(words, n_ri, protected_set)
            augmented.append(' '.join(a))

    # RS
    if alpha_rs > 0 and num_words > 1:
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a = random_swap(words, n_rs, protected_set)
            augmented.append(' '.join(a))

    # RD（只对非保护词尝试删除）
    if p_rd > 0 and num_words > 1:
        for _ in range(num_new_per_technique):
            a = random_deletion(words, p_rd, protected_set)
            augmented.append(' '.join(a))

    # 清洗+打乱
    augmented = [get_only_chars_vla(s, keep_digits=keep_digits, keep_hyphen=keep_hyphen) for s in augmented]
    shuffle(augmented)

    # 控制数量
    if num_aug >= 1:
        augmented = augmented[:num_aug]
    else:
        keep_prob = num_aug / max(1, len(augmented))
        augmented = [s for s in augmented if random.uniform(0,1) < keep_prob]

    return augmented
