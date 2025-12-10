from __future__ import annotations
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np

# FAISS
try:
    import faiss
except Exception:
    faiss = None

# Torch
import torch

# SentenceTransformer (MPNet)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# =======================
# Конфіг
# =======================
DATA_DIR = Path("MSRCaseStudy/ganttJSON")    # тут лежать methods.json, requirements.json, traces.json
CACHE_DIR = Path("emb_cache")

# 🔥 MPNet-модель через SentenceTransformer
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 256      # у SentenceTransformer напряму не використовується, але лишаємо для сумісності
RANDOM_SEED = 42
SIM_THRESHOLD = 0.5  # мінімальна схожість

# Для таблиці
K_VALUES = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000,3000,4000,5000]

for d in [DATA_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =======================
# Моделі даних
# =======================
@dataclass
class Method:
    id: int
    fullmethod: str
    classname: Optional[str] = None
    classid: Optional[int] = None
    sourcecode: Optional[str] = None

@dataclass
class Requirement:
    id: int
    requirementname: str
    text: Optional[str] = None

# =======================
# IO
# =======================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Файл не знайдено: {path}")
    with path.open("r", encoding="utf-8") as f:
        s = f.read().strip()
        return json.loads(s) if s else []

# =======================
# Парсери датасету
# =======================
def build_methods(methods_raw: List[Dict[str, Any]]) -> List[Method]:
    out = []
    for m in methods_raw or []:
        try:
            mid = int(m.get("id"))
        except Exception:
            continue
        classid = m.get("classid") or m.get("ownerclassid")
        try:
            classid = int(classid) if classid is not None else None
        except Exception:
            classid = None
        out.append(Method(
            id=mid,
            fullmethod=m.get("fullmethod") or m.get("methodname") or "",
            classname=m.get("classname") or None,
            classid=classid,
            sourcecode=m.get("method") or m.get("sourcecode") or ""
        ))
    # унікалізація по id
    return list({m.id: m for m in out}.values())

def build_requirements(reqs_raw: List[Dict[str, Any]]) -> List[Requirement]:
    out = []
    for r in reqs_raw or []:
        try:
            rid = int(r.get("id"))
        except Exception:
            continue
        text = r.get("text") or r.get("description") or "(немає тексту)"
        out.append(Requirement(
            id=rid,
            requirementname=r.get("requirementname") or f"R{rid}",
            text=str(text).strip()
        ))
    return list({r.id: r for r in out}.values())

def build_gold_truth(traces_raw: List[Dict[str, Any]]) -> Dict[str, Set[int]]:
    """
    Повертає: requirementname -> множина позитивних method_id
    """
    truth: Dict[str, Set[int]] = {}
    POS = {"T", "E", "1", "TRUE"}
    for t in traces_raw or []:
        req_name = str(t.get("requirement"))
        lab = str(t.get("goldfinal", t.get("label", "F"))).upper()
        if req_name and lab in POS:
            try:
                mid = int(t.get("methodid"))
            except Exception:
                continue
            truth.setdefault(req_name, set()).add(mid)
    return truth

# =======================
# Підготовка тексту (З УРАХУВАННЯМ КОНТЕКСТУ)
# =======================
def prepare_method_text(m: Method) -> str:
    """
    fullmethod + classname + sourcecode
    (структурний контекст).
    """
    return "\n".join(filter(None, [m.fullmethod, m.classname, m.sourcecode]))

def prepare_requirement_text(r: Requirement) -> str:
    return f"{r.requirementname}\n{r.text or ''}".strip()

# =======================
# MPNet embedder через SentenceTransformer (з fp16 на GPU)
# =======================
class MPNetEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        if SentenceTransformer is None:
            raise ImportError("Встановіть пакет 'sentence-transformers' для використання MPNet.")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Завантаження SentenceTransformer MPNet: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)

        # 🔥 fp16 тільки на GPU
        if self.device == "cuda":
            try:
                self.model = self.model.half()
                print("✅ Увімкнено fp16 (half precision) для MPNet на GPU")
            except Exception as e:
                print(f"⚠️ Не вдалося увімкнути fp16, працюємо у float32: {e}")

        print(f"✅ MPNet готовий, пристрій: {self.device}")

    def encode(self, texts: List[str], cache_file: Path) -> np.ndarray:
        print(f"🔹 MPNet-ембеддинг {len(texts)} елементів; кеш: {cache_file}")
        if cache_file.exists():
            print("⚡ Завантаження з кешу…")
            return np.load(cache_file)

        # SentenceTransformer сам ріже довжину, max_length явно не передаємо
        embs = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # одразу L2-нормалізація
        ).astype(np.float32)

        np.save(cache_file, embs)
        print(f"✅ Кеш збережено: {cache_file}")
        return embs

# =======================
# Індексатор (FAISS)
# =======================
class Indexer:
    def __init__(self, dim: int):
        if faiss is None:
            raise ImportError("Потрібно встановити faiss-cpu")
        self.id_map: List[int] = []
        self.index = faiss.IndexFlatIP(dim)

    def fit(self, vectors: np.ndarray, ids: List[int]):
        # на всяк випадок ще раз нормалізуємо
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.id_map = ids

    def search(self, q_vecs: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
        faiss.normalize_L2(q_vecs)
        D, I = self.index.search(q_vecs, k)
        out: List[List[Tuple[int, float]]] = []
        for i in range(len(q_vecs)):
            row = []
            for n, j in enumerate(I[i]):
                row.append((self.id_map[int(j)], float(D[i][n])))
            out.append(row)
        return out

# =======================
# Метрики для однієї вимоги
# =======================
def compute_metrics_for_single(pred_ids: List[int], gold_ids: Set[int]) -> Dict[str, float]:
    """
    pred_ids – список Top-K method_id (K = len(pred_ids))
    gold_ids – множина істинних method_id для цієї вимоги
    """
    if not pred_ids or not gold_ids:
        return {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "MAP": 0.0,
            "MRR": 0.0,
        }

    k = len(pred_ids)
    hits_total = sum(1 for p in pred_ids if p in gold_ids)

    precision = hits_total / k
    recall = hits_total / len(gold_ids)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # MRR
    mrr = 0.0
    for rank, mid in enumerate(pred_ids, start=1):
        if mid in gold_ids:
            mrr = 1.0 / rank
            break

    # AP: сума precision@i на релевантних позиціях / |gold_ids|
    ap_sum = 0.0
    rel_seen = 0
    for rank, mid in enumerate(pred_ids, start=1):
        if mid in gold_ids:
            rel_seen += 1
            ap_sum += rel_seen / rank
    map_ = ap_sum / len(gold_ids) if len(gold_ids) else 0.0

    topk_recall = 1.0 if hits_total > 0 else 0.0  # Hit@K

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MAP": float(map_),
        "MRR": float(mrr),
    }

# =======================
# Глобальна оцінка для різних K
# =======================
def main():
    print("📂 Завантаження датасету…")
    methods_raw = load_json(DATA_DIR / "methods.json")
    reqs_raw    = load_json(DATA_DIR / "requirements.json")
    traces_raw  = load_json(DATA_DIR / "traces.json")

    methods = build_methods(methods_raw)
    requirements = build_requirements(reqs_raw)
    gold_truth = build_gold_truth(traces_raw)

    print(f"Методів: {len(methods)}")
    print(f"Вимог:  {len(requirements)}")
    print(f"Вимог з gold-трасами: {sum(1 for r in requirements if r.requirementname in gold_truth)}")

    # --- ембеддинги (MPNet) ---
    embedder = MPNetEmbedder()
    method_texts = [prepare_method_text(m) for m in methods]
    req_texts    = [prepare_requirement_text(r) for r in requirements]

    model_tag = MODEL_NAME.split("/")[-1].replace("-", "_")
    methods_cache = CACHE_DIR / f"methods_{model_tag}_ctx.npy"
    reqs_cache    = CACHE_DIR / f"requirements_{model_tag}_ctx.npy"

    # m_emb = embedder.encode(method_texts, CACHE_DIR / "codebert_methods.npy")
    # r_emb = embedder.encode(req_texts, CACHE_DIR / "codebert_requirements.npy")

    m_emb = embedder.encode(method_texts, methods_cache)
    r_emb = embedder.encode(req_texts, reqs_cache)

    # --- індекс ---
    indexer = Indexer(m_emb.shape[1])
    indexer.fit(m_emb, [m.id for m in methods])

    # Шукаємо один раз з maxK
    max_k = max(K_VALUES)
    print(f"🔎 Пошук Top-{max_k} для всіх вимог…")
    all_retrieved = indexer.search(r_emb, max_k)  # список списків (для кожної вимоги)

    # мапимо: req_id -> [(method_id, score), ...]
    retrieved_map: Dict[int, List[Tuple[int,float]]] = {}
    for r, res in zip(requirements, all_retrieved):
        # залишаємо лише ті методи, де similarity >= поріг
        filtered = [(mid, score) for mid, score in res if score >= SIM_THRESHOLD]
        retrieved_map[r.id] = filtered

    # ======================
    # Обчислення таблиці
    # ======================
    print("\n==============================")
    print("   Залежність метрик від Top-K (MPNet)")
    print("==============================\n")

    header = [
        "K",
        "Precision",
        "Recall",
        "F1-score",
        "MAP",
        "MRR",
    ]
    print("\t".join(header))

    for K in K_VALUES:
        # сума по всіх вимогах для кожної колонки таблиці
        sum_metrics = {k: 0.0 for k in header[1:]}
        cnt = 0

        # відповідність назви колонки в таблиці → ключа у m (compute_metrics_for_single)
        metric_map = {
            "Precision":      "Precision",
            "Recall":         "Recall",
            "F1-score":       "F1",
            "MAP":            "MAP",
            "MRR":            "MRR",
        }

        for r in requirements:
            gold_ids = gold_truth.get(r.requirementname, set())
            if not gold_ids:
                continue

            preds = retrieved_map.get(r.id, [])[:K]
            pred_ids = [mid for (mid, _) in preds]
            if not pred_ids:
                continue

            m = compute_metrics_for_single(pred_ids, gold_ids)

            # акумулюємо метрики
            for col_name in sum_metrics:
                sum_metrics[col_name] += m[metric_map[col_name]]

            cnt += 1

        if cnt == 0:
            row = [K] + ["0.0000"] * (len(header) - 1)
        else:
            avg = {k: v / cnt for k, v in sum_metrics.items()}
            row = [
                K,
                f"{avg['Precision']:.4f}",
                f"{avg['Recall']:.4f}",
                f"{avg['F1-score']:.4f}",
                f"{avg['MAP']:.4f}",
                f"{avg['MRR']:.4f}",
            ]

        print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()
