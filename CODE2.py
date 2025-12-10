from __future__ import annotations
import os, json, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple

# --- базові ---
import numpy as np

# --- Tkinter / matplotlib (спершу бекенд, потім pyplot) ---
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Transformers / torch ---
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:
    AutoTokenizer = AutoModel = torch = None

# --- SentenceTransformers (SBERT) ---
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# --- FAISS / sklearn ---
try:
    import faiss
except Exception:
    faiss = None

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    NearestNeighbors = TfidfVectorizer = cosine_similarity = None

# --- joblib для кешу TF-IDF векторизатора ---
try:
    from joblib import dump as joblib_dump, load as joblib_load
except Exception:
    joblib_dump = joblib_load = None

# =======================
# Конфіг
# =======================
DATA_DIR = Path("MSRCaseStudy/ganttJSON")    # тут лежать methods.json, requirements.json, traces.json
CACHE_DIR = Path("emb_cache")

MODEL_NAME_CODEBERT = "microsoft/codebert-base"
MODEL_NAME_SBERT = "sentence-transformers/all-MiniLM-L6-v2"  
USE_FAISS = True
TOP_K = 500
MAX_LEN = 256
RANDOM_SEED = 42
SIM_THRESHOLD = 0  # мінімальна схожість

for d in [DATA_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
        return None
    with path.open("r", encoding="utf-8") as f:
        s = f.read().strip()
        return json.loads(s) if s else None

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
    # унікалізація за id
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
    (T / E  в goldfinal/label)
    """
    truth: Dict[str, Set[int]] = {}
    POS = {"T", "E",}
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
# Підготовка тексту
# =======================
def prepare_method_text(m: Method) -> str:
    return "\n".join(filter(None, [m.fullmethod, m.classname, m.sourcecode]))

def prepare_requirement_text(r: Requirement) -> str:
    return f"{r.requirementname}\n{r.text or ''}".strip()

# =======================
# Ембеддери
# =======================
class CodeBertEmbedder:
    def __init__(self, model_name=MODEL_NAME_CODEBERT, max_length=MAX_LEN, local_model_dir: Optional[str] = None):
        if AutoModel is None or torch is None:
            raise ImportError("Встановіть transformers та torch для CodeBERT")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.tokenizer, self.model = self._safe_load(model_name, local_model_dir)

        if self.device == "cuda":
            try:
                self.model.to(self.device, dtype=torch.float16)
            except Exception:
                self.model.to(self.device)
        else:
            self.model.to(self.device)
        self.model.eval()
        print(f"✅ CodeBERT готовий. Пристрій: {self.device}")

    def _safe_load(self, model_name: str, local_model_dir: Optional[str]):
        from transformers import AutoTokenizer, AutoModel

        if local_model_dir:
            print(f"⬇️ Завантаження CodeBERT з локальної папки: {local_model_dir}")
            tok = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True, local_files_only=True)
            mdl = AutoModel.from_pretrained(local_model_dir, local_files_only=True)
            return tok, mdl

        # 1) онлайн, стандартний кеш
        try:
            print("🧠 Завантаження CodeBERT (онлайн, стандартний кеш)...")
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            mdl = AutoModel.from_pretrained(model_name)
            return tok, mdl
        except Exception as e1:
            print(f"⚠️ Не вдалося (стандартний кеш): {e1}")


    def encode(self, texts: List[str], cache_file: Path) -> np.ndarray:
        print(f"🔹 CodeBERT embeddings: {len(texts)} елементів; кеш: {cache_file.name}")
        if cache_file.exists():
            print("⚡ Завантаження CodeBERT з кешу…")
            return np.load(cache_file)

        embs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**inputs)
                last_hidden = out.last_hidden_state
                mask = enc["attention_mask"].to(self.device).unsqueeze(-1)
                v = (last_hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                embs.append(v.detach().cpu().numpy())
            print(f"   🟢 {min(i + batch_size, len(texts))}/{len(texts)}")

        arr = np.vstack(embs).astype(np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        np.save(cache_file, arr)
        print(f"✅ Збережено CodeBERT кеш: {cache_file.name}")
        return arr

class SbertEmbedder:
    def __init__(self, model_name=MODEL_NAME_SBERT, max_length=MAX_LEN):
        if SentenceTransformer is None:
            raise ImportError("Встановіть sentence-transformers для SBERT (all-mpnet-base-v2)")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ SBERT ({model_name}) готовий. Пристрій: {device}")

    def encode(self, texts: List[str], cache_file: Path) -> np.ndarray:
        print(f"🔹 SBERT embeddings: {len(texts)} елементів; кеш: {cache_file.name}")
        if cache_file.exists():
            print("⚡ Завантаження SBERT з кешу…")
            return np.load(cache_file)
        truncated = [
            t[:self.max_length]  
            for t in texts
        ]
        embs = self.model.encode(
            truncated,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        np.save(cache_file, embs)
        print(f"✅ Збережено SBERT кеш: {cache_file.name}")
        return embs


# =======================
# Індексатор (для SBERT / CodeBERT)
# =======================
class Indexer:
    def __init__(self, dim, use_faiss=True):
        self.use_faiss = bool(use_faiss and (faiss is not None))
        if not self.use_faiss and NearestNeighbors is None:
            raise ImportError("Встановіть faiss-cpu або scikit-learn")
        self.id_map: List[int] = []
        self.index = faiss.IndexFlatIP(dim) if self.use_faiss else NearestNeighbors(metric="cosine")

    def fit(self, vectors: np.ndarray, ids: List[int]):
        if self.use_faiss:
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
        else:
            self.index.fit(vectors)
        self.id_map = ids

    def search(self, q_vecs: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
        if self.use_faiss:
            faiss.normalize_L2(q_vecs)
            D, I = self.index.search(q_vecs, k)
            return [[(self.id_map[j], float(D[i][n])) for n, j in enumerate(I[i])] for i in range(len(q_vecs))]
        else:
            dist, ind = self.index.kneighbors(q_vecs, n_neighbors=k)
            return [[(self.id_map[j], 1.0 - float(dist[i][n])) for n, j in enumerate(ind[i])] for i in range(len(q_vecs))]

# =======================
# Метрики (локально для однієї вимоги)
# =======================
def compute_local_metrics(pred_ids: List[int], gold_ids: Set[int]) -> Dict[str, float]:
   
    if not pred_ids or not gold_ids:
        return {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "MAP": 0.0,
            "MRR": 0.0,
        }

    hits_total = sum(1 for p in pred_ids if p in gold_ids)
    precision = hits_total / len(pred_ids)
    recall = hits_total / len(gold_ids)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # MRR
    mrr = 0.0
    for i, pid in enumerate(pred_ids, 1):
        if pid in gold_ids:
            mrr = 1.0 / i
            break

    # AP
    ap_sum, rel_seen = 0.0, 0
    for i, pid in enumerate(pred_ids, 1):
        if pid in gold_ids:
            rel_seen += 1
            ap_sum += rel_seen / i
    map_ = ap_sum / len(gold_ids) if len(gold_ids) else 0.0


    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MAP": float(map_),
        "MRR": float(mrr),
    }

# =======================
# TF-IDF pipeline
# =======================
def run_tfidf_pipeline(methods: List[Method],
                       requirements: List[Requirement],
                       top_k: int) -> Dict[int, List[Tuple[int, float]]]:
    if TfidfVectorizer is None or cosine_similarity is None:
        raise ImportError("Для TF-IDF потрібен scikit-learn")

    print("\n==========================")
    print("   МОДЕЛЬ: TF-IDF")
    print("==========================")

    method_texts = [prepare_method_text(m) for m in methods]
    req_texts = [prepare_requirement_text(r) for r in requirements]

    vec_path = CACHE_DIR / "tfidf_vectorizer.joblib"

    if vec_path.exists() and joblib_load is not None:
        print("⚡ Завантаження TF-IDF векторизатора з кешу…")
        vectorizer: TfidfVectorizer = joblib_load(vec_path)
    else:
        print("🧠 Навчання TF-IDF векторизатора…")
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            lowercase=True
        )
        vectorizer.fit(method_texts + req_texts)
        if joblib_dump is not None:
            joblib_dump(vectorizer, vec_path)
            print(f"✅ Збережено TF-IDF векторизатор у {vec_path.name}")
        else:
            print("⚠️ joblib недоступний – векторизатор не буде закешовано.")

    print("🔹 Обчислення TF-IDF матриць…")
    M = vectorizer.transform(method_texts)   # (n_methods, d)
    R = vectorizer.transform(req_texts)      # (n_reqs, d)

    print("🔹 Обчислення косинусної схожості TF-IDF (може бути повільно)…")
    sims = cosine_similarity(R, M)           # (n_reqs, n_methods)

    retrieved: Dict[int, List[Tuple[int, float]]] = {}
    n_methods = sims.shape[1]

    for i, r in enumerate(requirements):
        row = sims[i]
        k = min(top_k, n_methods)
        if k <= 0:
            retrieved[r.id] = []
            continue
        # індекси топ-k
        top_idx = np.argpartition(row, -k)[-k:]
        top_idx_sorted = top_idx[np.argsort(-row[top_idx])]
        retrieved[r.id] = [(methods[j].id, float(row[j])) for j in top_idx_sorted if row[j] >= SIM_THRESHOLD]


    print("✅ TF-IDF: побудовано Top-K списки для всіх вимог.")
    return retrieved

# =======================
# SBERT pipeline
# =======================
def run_sbert_pipeline(methods: List[Method],
                       requirements: List[Requirement],
                       top_k: int) -> Dict[int, List[Tuple[int, float]]]:
    print("\n==========================")
    print("   МОДЕЛЬ: SBERT (all-MiniLM-L6-v2)")
    print("==========================")

    embedder = SbertEmbedder(MODEL_NAME_SBERT)
    method_texts = [prepare_method_text(m) for m in methods]
    req_texts = [prepare_requirement_text(r) for r in requirements]

    m_emb = embedder.encode(method_texts, CACHE_DIR / "sbert_methods.npy")
    r_emb = embedder.encode(req_texts, CACHE_DIR / "sbert_requirements.npy")

    indexer = Indexer(m_emb.shape[1], USE_FAISS)
    indexer.fit(m_emb, [m.id for m in methods])
    lists = indexer.search(r_emb, top_k)

    # Вставляємо фільтр за порогом
    filtered_lists = []
    for res in lists:
        filtered = [(mid, score) for mid, score in res if score >= SIM_THRESHOLD]
        filtered_lists.append(filtered)

    retrieved = {r.id: res for r, res in zip(requirements, filtered_lists)}

    print("✅ SBERT: побудовано Top-K списки для всіх вимог.")
    return retrieved

# =======================
# CodeBERT pipeline
# =======================
def run_codebert_pipeline(methods: List[Method],
                          requirements: List[Requirement],
                          top_k: int) -> Dict[int, List[Tuple[int, float]]]:
    print("\n==========================")
    print("   МОДЕЛЬ: CodeBERT")
    print("==========================")

    embedder = CodeBertEmbedder()
    method_texts = [prepare_method_text(m) for m in methods]
    req_texts = [prepare_requirement_text(r) for r in requirements]

    m_emb = embedder.encode(method_texts, CACHE_DIR / "codebert_methods.npy")
    r_emb = embedder.encode(req_texts, CACHE_DIR / "codebert_requirements.npy")

    indexer = Indexer(m_emb.shape[1], USE_FAISS)
    indexer.fit(m_emb, [m.id for m in methods])
    lists = indexer.search(r_emb, top_k)

    # Фільтр за порогом
    filtered_lists = []
    for res in lists:
        filtered = [(mid, score) for mid, score in res if score >= SIM_THRESHOLD]
        filtered_lists.append(filtered)

    retrieved = {r.id: res for r, res in zip(requirements, filtered_lists)}

    print("✅ CodeBERT: побудовано Top-K списки для всіх вимог.")
    return retrieved

# =======================
# Глобальні метрики по моделі
# =======================
def compute_global_metrics_for_model(
    model_name: str,
    retrieved_map: Dict[int, List[Tuple[int, float]]],
    requirements: List[Requirement],
    gold_truth: Dict[str, Set[int]],
    top_k: int
) -> Dict[str, float]:
    sum_metrics = {
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
        "MAP": 0.0,
        "MRR": 0.0,
    }
    cnt = 0

    for r in requirements:
        gold_ids = gold_truth.get(r.requirementname, set())
        pred_pairs = retrieved_map.get(r.id, [])
        pred_ids = [mid for (mid, _) in pred_pairs[:top_k]]

        if not pred_ids or not gold_ids:
            continue

        m = compute_local_metrics(pred_ids, gold_ids)
        for k in sum_metrics:
            sum_metrics[k] += m[k]
        cnt += 1

    if cnt == 0:
        print(f"❌ {model_name}: немає вимог з gold-трасами.")
        return {k: 0.0 for k in sum_metrics}

    avg = {k: v / cnt for k, v in sum_metrics.items()}

    print(f"\n📊 Підсумкові метрики для моделі {model_name} (усереднено по {cnt} вимогах, Top-{top_k}):")
    for k, v in avg.items():
        print(f"  {k:12s} = {v:.4f}")

    return avg

# =======================
# Повний пайплайн для однієї моделі
# =======================
def run_model_pipeline(
    model_name: str,
    methods: List[Method],
    requirements: List[Requirement],
    gold_truth: Dict[str, Set[int]],
    top_k: int
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[str, float]]:

    if model_name == "TF-IDF":
        retrieved = run_tfidf_pipeline(methods, requirements, top_k)
    elif model_name == "SBERT":
        retrieved = run_sbert_pipeline(methods, requirements, top_k)
    elif model_name == "CodeBERT":
        retrieved = run_codebert_pipeline(methods, requirements, top_k)
    else:
        raise ValueError(f"Невідома модель: {model_name}")

    avg_metrics = compute_global_metrics_for_model(model_name, retrieved, requirements, gold_truth, top_k)
    return retrieved, avg_metrics

# =======================
# Завантаження датасету (спільне для всіх моделей)
# =======================
def load_dataset(data_dir: Path):
    data = {k: load_json(data_dir / f"{k}.json") for k in ["methods", "requirements", "traces"]}
    if not all(data.values()):
        miss = [k for k, v in data.items() if not v]
        raise FileNotFoundError(f"Відсутні файли: {miss}")
    methods = build_methods(data["methods"])
    requirements = build_requirements(data["requirements"])
    gold_truth = build_gold_truth(data["traces"])
    print(f"✅ Завантажено датасет: methods={len(methods)}, requirements={len(requirements)}, traces={len(data['traces'])}")
    return methods, requirements, gold_truth, data["traces"]

# =======================
# GUI (із перемикачем моделей)
# =======================
def launch_gui(
    requirements: List[Requirement],
    methods: List[Method],
    gold_truth: Dict[str, Set[int]],
    traces: List[Dict[str, Any]],
    retrieved_by_model: Dict[str, Dict[int, List[Tuple[int, float]]]],
    top_k_default: int = TOP_K
):
    req_dict = {r.id: r for r in requirements}
    meth_dict = {m.id: m for m in methods}
    POSITIVE_LABELS = {"T", "E", "1", "TRUE"}

    def is_positive(t: dict) -> bool:
        lab = str(t.get("goldfinal", t.get("label", "F"))).upper()
        return lab in POSITIVE_LABELS

    def get_gold_for_req(req_name: str):
        gold_traces, gold_mids = [], set()
        for t in traces:
            if t.get("requirement") == req_name and is_positive(t):
                try:
                    mid = int(t.get("methodid"))
                except Exception:
                    continue
                gold_traces.append(t)
                gold_mids.add(mid)
        return gold_traces, gold_mids

    # --- Вікно ---
    root = tk.Tk()
    root.title("🔎 Traceability GUI (TF-IDF / SBERT / CodeBERT + GOLD + метрики)")
    # одразу на весь екран
    try:
        root.state("zoomed")    # Windows
    except Exception:
        root.attributes("-zoomed", True)  # Linux/macOS
    root.minsize(1200, 720)

    # Верхня панель (GRID)
    control = ttk.Frame(root, padding=10)
    control.grid(row=0, column=0, sticky="ew")
    for c in range(9):
        control.grid_columnconfigure(c, weight=(0 if c in (0, 2, 4, 6) else 1))

    ttk.Label(control, text="ID вимоги:").grid(row=0, column=0, sticky="w")
    entry_id = ttk.Entry(control, width=10)
    entry_id.grid(row=0, column=1, sticky="w", padx=(6, 12))

    ttk.Label(control, text="Top-K:").grid(row=0, column=2, sticky="w")
    entry_topk = ttk.Entry(control, width=7)
    entry_topk.insert(0, str(top_k_default))
    entry_topk.grid(row=0, column=3, sticky="w", padx=(6, 12))

    ttk.Label(control, text="Модель:").grid(row=0, column=4, sticky="w")
    model_var = tk.StringVar(value="CodeBERT")
    combo_model = ttk.Combobox(
        control,
        textvariable=model_var,
        values=list(retrieved_by_model.keys()),
        state="readonly",
        width=14
    )
    combo_model.grid(row=0, column=5, sticky="w", padx=(6, 12))

    btn_show = ttk.Button(control, text="Показати вимогу")
    btn_show.grid(row=0, column=6, sticky="w")

    btn_global = ttk.Button(control, text="Глобальні метрики")
    btn_global.grid(row=0, column=7, sticky="w", padx=(10, 0))

    status_lbl = ttk.Label(control, text="", foreground="#555")
    status_lbl.grid(row=0, column=8, sticky="w", padx=(12, 0))

    # Центральний спліт
    paned = ttk.Panedwindow(root, orient="horizontal")
    paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 6))

    left_frame = ttk.Frame(paned)
    right_frame = ttk.Frame(paned)
    paned.add(left_frame, weight=1)
    paned.add(right_frame, weight=1)

    left_txt = scrolledtext.ScrolledText(left_frame, width=80, height=25, wrap="word")
    left_txt.pack(fill="both", expand=True, padx=(0, 6), pady=0)

    right_txt = scrolledtext.ScrolledText(right_frame, width=80, height=25, wrap="word")
    right_txt.pack(fill="both", expand=True, padx=(6, 0), pady=0)

    # Деталі
    detailed_txt = scrolledtext.ScrolledText(root, width=140, height=8, wrap="word")
    detailed_txt.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 6))

    # Полотно для графіка
    chart_frame = ttk.Frame(root)
    chart_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))

    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    # адаптивність
    root.grid_rowconfigure(1, weight=2)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=2)
    root.grid_columnconfigure(0, weight=1)

    # --- логіка показу для однієї вимоги ---
    def show_methods():
        try:
            rid = int(entry_id.get().strip())
        except Exception:
            messagebox.showerror("Помилка", "Введи ціле число для ID вимоги.")
            return
        try:
            top_k = max(1, int(entry_topk.get().strip()))
        except Exception:
            messagebox.showerror("Помилка", "Введи ціле число для Top-K.")
            return

        model_name = model_var.get()
        if model_name not in retrieved_by_model:
            messagebox.showerror("Помилка", f"Невідома модель: {model_name}")
            return

        retrieved_map = retrieved_by_model[model_name]

        req = req_dict.get(rid)
        if not req:
            messagebox.showerror("Помилка", f"Вимога {rid} не знайдена")
            return

        # очистка
        for w in (left_txt, right_txt, detailed_txt):
            w.config(state="normal")
            w.delete("1.0", tk.END)

        gold_traces, gold_method_ids = get_gold_for_req(req.requirementname)

        # Ліва колонка: Top-K
        left_txt.insert(tk.END, f"🔹 Модель: {model_name}\n")
        left_txt.insert(tk.END, f"🔹 Вимога {req.requirementname} (ID={req.id}):\n{req.text}\n\n")
        left_txt.insert(tk.END, f"📊 Передбачені методи (Top-{top_k}):\n")
        preds_for_req = retrieved_map.get(rid, [])[:top_k]
        for i, (mid, score) in enumerate(preds_for_req, 1):
            m = meth_dict.get(mid)
            name = (m.fullmethod.split('.')[-1] if (m and m.fullmethod) else str(mid))
            # trace_id лише якщо є позитивна gold пара для цього methodid
            t = next((t for t in gold_traces if str(t.get("methodid")) == str(mid)), None)
            trace_id = t.get("id") if t else None
            if trace_id is not None:
                left_txt.insert(tk.END, f"{i}. method_id:{mid} trace_id:{trace_id} {name}  [{score:.4f}]\n")
            else:
                left_txt.insert(tk.END, f"{i}. method_id:{mid} {name}  [{score:.4f}]\n")

        # Права колонка: тільки GOLD
        right_txt.insert(tk.END, "📘 GOLD (позитивні треси для цієї вимоги):\n")
        if not gold_traces:
            right_txt.insert(tk.END, "— немає позитивних розміток —\n")
        else:
            right_txt.insert(tk.END, f"(усього: {len(gold_traces)})\n")
            for t in sorted(gold_traces, key=lambda x: int(x.get("methodid", 0))):
                mid = int(t.get("methodid"))
                m = meth_dict.get(mid)
                name = (m.fullmethod.split('.')[-1] if (m and m.fullmethod) else str(mid))
                lab = str(t.get('goldfinal', t.get('label', ''))).upper()
                right_txt.insert(
                    tk.END,
                    f"req:{req.requirementname} | trace_id:{t.get('id')} | methodid:{mid} | {name} | label:{lab}\n"
                )

        # Деталі + метрики
        pred_method_ids = [mid for (mid, _) in preds_for_req]
        hits = sum(1 for pid in pred_method_ids if pid in gold_method_ids)
        precision_pct = (hits / len(pred_method_ids) * 100.0) if pred_method_ids else 0.0

        detailed_txt.insert(tk.END, f"📝 Детальний вивід для {req.requirementname} (модель {model_name}, Top-{top_k}):\n")
        for i, (mid, score) in enumerate(preds_for_req, 1):
            m = meth_dict.get(mid)
            name = (m.fullmethod.split('.')[-1] if (m and m.fullmethod) else str(mid))
            is_gold = "✓" if mid in gold_method_ids else " "
            detailed_txt.insert(
                tk.END,
                f"{i:>2}. [{is_gold}] req:{req.requirementname}  method_id:{mid}  {name}  [score={score:.4f}]\n"
            )
        detailed_txt.insert(
            tk.END,
            f"\n✅ Збігів: {hits} / {len(pred_method_ids)} | Точність (по Top-{top_k}): {precision_pct:.2f}%\n"
        )

        # Локальні метрики
        metrics = compute_local_metrics(pred_method_ids, gold_method_ids)
        for k, v in metrics.items():
            detailed_txt.insert(tk.END, f"  {k}: {v:.4f}\n")

        # Графік
        ax.clear()
        keys = list(metrics.keys())
        vals = [metrics[k] for k in keys]
        bars = ax.bar(keys, vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(f"Метрики для {req.requirementname} (модель {model_name})")
        for b, val in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for label in ax.get_xticklabels():
            label.set_rotation(12)
            label.set_ha("right")
        canvas.draw()

        # статус
        if not preds_for_req:
            status_lbl.config(text="Немає кандидатів для цієї вимоги.")
        elif not gold_method_ids:
            status_lbl.config(text="У GOLD немає позитивів для цієї вимоги.")
        else:
            status_lbl.config(text=f"Модель: {model_name}")

    # --- глобальні (середні) метрики по всіх вимогах для обраної моделі ---
    def show_global_metrics():
        for w in (left_txt, right_txt, detailed_txt):
            w.config(state="normal")
            w.delete("1.0", tk.END)

        model_name = model_var.get()
        if model_name not in retrieved_by_model:
            detailed_txt.insert(tk.END, f"❌ Невідома модель: {model_name}\n")
            status_lbl.config(text="Помилка моделі.")
            return

        retrieved_map = retrieved_by_model[model_name]

        try:
            top_k = max(1, int(entry_topk.get().strip()))
        except Exception:
            top_k = top_k_default

        sum_metrics = {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "MAP": 0.0,
            "MRR": 0.0,
        }
        cnt = 0

        for r in requirements:
            gold_ids = gold_truth.get(r.requirementname, set())
            pred_pairs = retrieved_map.get(r.id, [])
            pred_ids = [mid for (mid, _) in pred_pairs[:top_k]]
            if not pred_ids or not gold_ids:
                continue
            m = compute_local_metrics(pred_ids, gold_ids)
            for k in sum_metrics:
                sum_metrics[k] += m[k]
            cnt += 1

        if cnt == 0:
            detailed_txt.insert(tk.END, "❌ Немає вимог з gold-трасами для обчислення глобальних метрик.\n")
            status_lbl.config(text="Немає даних для глобальних метрик.")
            return

        avg = {k: v / cnt for k, v in sum_metrics.items()}

        detailed_txt.insert(
            tk.END,
            f"📊 Глобальні (усереднені) метрики по всіх вимогах для моделі {model_name} (Top-{top_k}):\n"
        )
        detailed_txt.insert(tk.END, f"Кількість вимог з gold-трасами: {cnt}\n\n")
        for k, v in avg.items():
            detailed_txt.insert(tk.END, f"{k}: {v:.4f}\n")

        # графік
        ax.clear()
        keys = list(avg.keys())
        vals = [avg[k] for k in keys]
        bars = ax.bar(keys, vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(f"Глобальні метрики (модель {model_name})")
        for b, val in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for label in ax.get_xticklabels():
            label.set_rotation(12)
            label.set_ha("right")
        canvas.draw()

        status_lbl.config(text=f"Глобальні метрики обчислено (модель {model_name}).")

    btn_show.configure(command=show_methods)
    btn_global.configure(command=show_global_metrics)

    # коректне закриття
    def on_close():
        try:
            canvas_widget.destroy()
        except Exception:
            pass
        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            root.quit()
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda e: on_close())
    root.mainloop()

# =======================
# Main
# =======================
if __name__ == "__main__":
    if faiss is None:
        USE_FAISS = False

    methods, requirements, gold_truth, traces = load_dataset(DATA_DIR)

    retrieved_by_model: Dict[str, Dict[int, List[Tuple[int, float]]]] = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    # Порядок: TF-IDF → SBERT → CodeBERT
    for model_name in ["TF-IDF", "SBERT", "CodeBERT"]:
        try:
            retrieved, avg_metrics = run_model_pipeline(
                model_name,
                methods,
                requirements,
                gold_truth,
                TOP_K
            )
            retrieved_by_model[model_name] = retrieved
            metrics_by_model[model_name] = avg_metrics
        except ImportError as e:
            print(f"\n❌ Пропускаємо модель {model_name}: {e}")
        except Exception as e:
            print(f"\n❌ Помилка під час виконання моделі {model_name}: {e}")

    # Коротке резюме для Таблиці 4.1
    print("\n==========================")
    print("   ПІДСУМКОВА ТАБЛИЦЯ (для магістерської)")
    print("==========================")
    header = f"{'Модель':10s} | {'Precision':9s} | {'Recall':7s} | {'F1-score':8s} | {'MAP':6s} | {'MRR':6s}"
    print(header)
    print("-" * len(header))
    for model_name in ["TF-IDF", "SBERT", "CodeBERT"]:
        m = metrics_by_model.get(model_name)
        if not m:
            continue
        print(
            f"{model_name:10s} | "
            f"{m['Precision']:9.4f} | "
            f"{m['Recall']:7.4f} | "
            f"{m['F1']:8.4f} | "
            f"{m['MAP']:6.4f} | "
            f"{m['MRR']:6.4f} "
        )

    # Запуск GUI з перемикачем моделей
    if retrieved_by_model:
        launch_gui(requirements, methods, gold_truth, traces, retrieved_by_model, TOP_K)
    else:
        print("❌ Жодна модель не була успішно побудована – GUI не запускається.")
