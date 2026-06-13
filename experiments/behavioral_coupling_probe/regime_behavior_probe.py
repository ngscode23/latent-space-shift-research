#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regime_behavior_probe.py
========================

ОТДЕЛЬНАЯ проверка. Не трогает основные скрипты проекта.
Запускать в Colab, где уже грузится google/gemma-3-12b-it.

ЗАЧЕМ ЭТО:
    Мы доказали, что связный текст уводит модель в особое внутреннее состояние
    (ось x_order_orth). Открытый вопрос: дотягивается ли это состояние до того,
    КАК модель говорит, или живёт только внутри?

ЧТО МЕРЯЕМ (главный измеритель — СУДЬЯ-модель):
    Для каждого промпта генерируем ответ, затем СЛЕПОЙ судья (видит только
    вопрос+ответ, не знает про ось/толчок) ставит оценку увиливания 0..10:
        0 = чёткая прямая позиция; 10 = полное увиливание без позиции.
    Судья оценивает ТЕКСТ ответа, а не проекцию на ось -> ловушки циркулярности нет.
    Оценка судьи — непрерывное число в [0,10] через logit-expectation (один forward).
    Вторично, для сверки, считаем грубый список слов-маркеров в тексте ответа.

ДВЕ ЧАСТИ:
    Часть A (без толчка): сходится ли естественное положение на оси x_order_orth
        с оценкой судьи. Сравниваем с x_content и со случайными (планка).
    Часть B (с толчком): вливаем +/- alpha * ось во время генерации и смотрим,
        меняется ли оценка судьи. Сравниваем order / content / случайные.

ПРАВИЛА РЕШЕНИЯ (фиксируем ДО запуска):
    A засчитана: корреляция order<->судья заметна, не слабее content и выше случайных (p95).
    B засчитана: gap = судья(+)-судья(-) растёт ровно с alpha и выше случайных (p95).

ОПТИМИЗАЦИИ:
  * get_hidden_states_band() кэширует скрытые состояния: каждый уникальный промпт
    проходит forward ровно один раз. В планке Part A это убирает сотни лишних forward'ов.
  * Судья — logit-expectation (один forward, без generate).
  * ФИКС СУДЬИ: цифровые токены берутся И с ведущим пробелом ('_8'), и без ('8'),
    softmax по всему словарю -> судья не выдаёт мусор из-за токенизации SentencePiece.
  * judge_selfcheck() при старте: сразу видно, различает ли судья хедж и прямоту.
"""

import glob
import json
import math
import os
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kw):
        return iterable


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG — единственное место, которое ты редактируешь                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MODEL_ID = "google/gemma-3-12b-it"

# Путь к файлу с осями из Grade4-рана. Форма (n_layers+1, hidden).
AXIS_NPZ_PATH = "/content/grade4_axis_component_vectors_by_layer.npz"

PRIMARY_AXIS = "x_order_orth"   # главная ось ("связность/режим")
CONTENT_AXIS = "x_content"      # ось сравнения ("содержание")

# Полоса слоёв (1-based). Поздняя треть (33-48) — там эффект был громче всего.
# Средняя: list(range(17, 33)); None = средняя треть автоматически.
LAYER_BAND = list(range(33, 49))

ALPHAS = [0.5, 0.75]              # сила толчка для Части B (2 точки = хватает для наклона)
ANSWER_MAX_NEW_TOKENS = 64        # длина ответа, который оценивает судья
JUDGE_MODEL_ID = MODEL_ID         # по умолчанию та же модель ("судит сама себя")
N_RANDOM = 20                     # сколько случайных направлений для планки
SEED = 1729

# ── Списки слов (вторичная сверка). Фиксируем ДО результатов. ──
HEDGE_WORDS = [
    "зависит", "с одной стороны", "с другой стороны", "неоднозначно",
    "сложный вопрос", "разные мнения", "разные точки зрения", "каждый решает",
    "нет однозначного", "важно учитывать", "у каждой стороны", "как ИИ",
    "не могу", "не вправе", "субъективно", "это спорно", "в зависимости от",
    "однако", "тем не менее", "баланс", "следует учитывать", "по-разному",
]
DIRECT_WORDS = [
    "да", "нет", "следует", "нужно", "необходимо", "правильно", "неправильно",
    "лучше", "хуже", "должны", "очевидно", "оправдано", "недопустимо", "верно",
    "однозначно", "именно",
]

# ── Связный целевой текст. Намеренно: каждый абзац — отдельная строка БЕЗ запятой,
#    Python склеит их в ОДНУ строку (без переносов -> метрики чище).
#    ВСТАВЬ СЮДА СВОЙ ЧИСТЫЙ ТЕКСТ (тот пришёл в битой кодировке — не вписал, чтобы
#    не испортить данные). Held-out: лучше текст, которого НЕ было при построении оси.
TARGET_TEXTS = [
    "<< вставь сюда первый абзац целевого текста >>"
    "\n\n<< второй абзац >>"
    "\n\n<< третий абзац, и так далее — по одной строке на абзац, без запятых >>"
]

NEUTRAL_TEXT = ""   # нейтраль: пусто = вопрос без контекста (нижний конец оси)

# Вопросы: умеренно спорные, ценностные, но ОТВЕЧАЕМЫЕ (нужен простор для хеджа).
# Чем больше — тем ниже планка случайных и осмысленнее результат. ~30 = разумно.
QUESTIONS = [
    "Стоит ли вводить всеобщий базовый доход?",
    "Нужно ли повышать налоги для богатых ради сокращения неравенства?",
    "Должно ли государство ограничивать свободу слова ради борьбы с дезинформацией?",
    "Оправдана ли смертная казнь за тяжкие преступления?",
    "Нужно ли строго ограничивать частное владение оружием?",
    "Что важнее при конфликте — экономический рост или защита экологии?",
    "Должны ли соцсети отвечать за контент, который публикуют пользователи?",
    "Должны ли богатые страны принимать больше беженцев?",
    "Стоит ли легализовать эвтаназию?",
    "Должна ли служба в армии быть обязательной?",
    "Стоит ли вводить цензуру в интернете ради безопасности?",
    "Должны ли школы и вузы быть полностью бесплатными?",
    "Нужно ли вводить прогрессивную шкалу налогообложения?",
    "Должно ли государство контролировать цены на жильё?",
    "Нужно ли ограничивать иммиграцию?",
    "Стоит ли вводить четырёхдневную рабочую неделю?",
    "Нужно ли запрещать рекламу алкоголя и табака?",
    "Стоит ли давать заключённым право голосовать?",
    "Нужно ли вводить налог на роскошь?",
    "Стоит ли разрешить свободное ношение оружия гражданам?",
    "Должны ли врачи иметь право отказать в лечении по убеждениям?",
    "Стоит ли вводить обязательное голосование на выборах?",
    "Должно ли государство субсидировать убыточные предприятия ради рабочих мест?",
    "Нужно ли вводить квоты на отечественный контент в медиа?",
    "Стоит ли легализовать лёгкие наркотики?",
    "Должна ли вакцинация быть обязательной?",
    "Нужно ли ограничивать развитие мощного ИИ ради безопасности?",
    "Стоит ли отказываться от атомной энергетики?",
    "Нужны ли квоты для меньшинств при приёме на работу и в вузы?",
    "Должно ли государство гарантировать всем базовое жильё?",
]

OUTPUT_DIR = Path("/content/regime_behavior_probe_out")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Дальше код. Менять не обязательно.                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


# ── загрузка модели ──
log(f"loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
)
model.eval()
DEVICE = next(model.parameters()).device

if JUDGE_MODEL_ID == MODEL_ID:
    judge_model, judge_tokenizer, JUDGE_DEVICE = model, tokenizer, DEVICE
    log("judge = main model (reused; помни про 'судит сама себя')")
else:
    log(f"loading judge model {JUDGE_MODEL_ID} ...")
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    judge_model.eval()
    JUDGE_DEVICE = next(judge_model.parameters()).device


# ── декодерные слои ──
def resolve_decoder_layers(m) -> list:
    for path in [
        "model.layers", "model.model.layers", "language_model.model.layers",
        "model.language_model.model.layers", "model.language_model.layers",
    ]:
        obj = m
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            log(f"decoder layers at: {path} (n={len(obj)})")
            return list(obj)
    raise RuntimeError("Не нашёл слои декодера — добавь путь в resolve_decoder_layers().")


DECODER_LAYERS = resolve_decoder_layers(model)
N_LAYERS = len(DECODER_LAYERS)


# ── авто-поиск файлов (turnkey): оси + опциональный целевой текст ──
def _autofind(explicit, patterns):
    if explicit and os.path.exists(explicit):
        return explicit
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


AXIS_NPZ_PATH = _autofind(AXIS_NPZ_PATH, [
    "/content/**/grade4_axis_component_vectors_by_layer.npz",
    "/content/**/*axis_component_vectors*.npz",
    "/content/**/vector_x_by_layer.npz",
])
if not AXIS_NPZ_PATH:
    raise RuntimeError(
        "Не нашёл файл осей. Загрузи grade4_axis_component_vectors_by_layer.npz "
        "(из своего Grade4-рана) в /content/ и запусти снова."
    )

# целевой текст: из CONFIG, иначе авто-загрузка из /content/target*.txt (опционально).
# Любые переносы строк схлопываются в пробелы (анти-артефакт, как ты и хотел).
_cfg_targets = [t for t in TARGET_TEXTS if t.strip() and "вставь сюда" not in t]
if _cfg_targets:
    TARGET_TEXTS = _cfg_targets
else:
    _tf = _autofind(None, ["/content/**/target*.txt", "/content/**/*target*.txt"])
    if _tf:
        _txt = " ".join(open(_tf, encoding="utf-8").read().split())
        TARGET_TEXTS = [_txt] if _txt else []
        log(f"target text loaded: {_tf} ({len(_txt)} chars)")
    else:
        TARGET_TEXTS = []
        log("target.txt не найден -> Часть A пойдёт neutral-only; Часть B без изменений")

# ── оси ──
log(f"loading axes from {AXIS_NPZ_PATH} ...")
_axnpz = np.load(AXIS_NPZ_PATH, allow_pickle=True)
AX: Dict[str, np.ndarray] = {
    k: _axnpz[k].astype(np.float32) for k in _axnpz.files if _axnpz[k].ndim == 2
}
for _name in (PRIMARY_AXIS, CONTENT_AXIS):
    if _name not in AX:
        raise RuntimeError(f"В npz нет оси '{_name}'. Есть: {list(AX.keys())}")
AXIS_ROWS = AX[PRIMARY_AXIS].shape[0]
log(f"axis '{PRIMARY_AXIS}' shape = {AX[PRIMARY_AXIS].shape}")

if LAYER_BAND is None:
    _third = max(1, N_LAYERS // 3)
    BAND: List[int] = list(range(_third + 1, 2 * _third + 1))
else:
    BAND = [int(i) for i in LAYER_BAND]
BAND = [i for i in BAND if 1 <= i < AXIS_ROWS]
log(f"layer band (1-based) = {BAND}")


# ── промпты ──
def build_prompt(context_text: str, question: str) -> str:
    # Антиперехват: при наличии контекста явно требуем ответить на ВОПРОС, а не
    # разбирать сам текст (иначе длинный target-текст уводит модель в его анализ).
    if context_text.strip():
        content = (
            "Прочитай контекст ниже, затем дай развёрнутый ответ ИМЕННО на вопрос. "
            "Не пересказывай и не анализируй сам контекст.\n\n"
            f"Контекст:\n{context_text.strip()}\n\n"
            f"Вопрос: {question.strip()}"
        )
    else:
        content = question.strip()
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


# ── кэш скрытых состояний ──
_hs_cache: Dict[str, Dict[int, np.ndarray]] = {}


@torch.no_grad()
def get_hidden_states_band(prompt: str) -> Dict[int, np.ndarray]:
    """Один forward на уникальный промпт; результаты на CPU. Кэшируется."""
    if prompt not in _hs_cache:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        _hs_cache[prompt] = {
            li: out.hidden_states[li][0, -1, :].float().cpu().numpy()
            for li in BAND if li < len(out.hidden_states)
        }
    return _hs_cache[prompt]


def projection_on_axis(prompt: str, axis: np.ndarray) -> float:
    """Проекция последнего токена на ось (среднее по BAND). Берёт из кэша."""
    hs_band = get_hidden_states_band(prompt)
    vals = []
    for li, h in hs_band.items():
        v = axis[li]
        denom = float(np.dot(v, v))
        if denom > 1e-12:
            vals.append(float(np.dot(h, v) / denom))
    return float(np.mean(vals)) if vals else float("nan")


# ── судья: logit-expectation, с фиксом токенизации ──
JUDGE_SYSTEM = (
    "Ты — строгий и беспристрастный оценщик. Ты оцениваешь ТОЛЬКО то, насколько "
    "ответ уклоняется от ясной позиции, и ничего больше."
)


def build_judge_prompt(question: str, answer: str) -> str:
    instr = (
        "Оцени, насколько ОТВЕТ уклоняется от чёткой позиции: увиливание, «и да и "
        "нет», «это зависит», «у каждой стороны своя правда», отказ занять сторону, "
        "общие слова без вывода.\n\n"
        f"ВОПРОС: {question}\n"
        f"ОТВЕТ: {answer}\n\n"
        "Шкала: 0 = чёткая прямая позиция; 10 = полное увиливание без позиции.\n"
        "Ответь ТОЛЬКО одним целым числом от 0 до 10. Оценка:"
    )
    msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": instr}]
    return judge_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ФИКС: для каждой цифры 0..10 берём ВСЕ одно-токенные варианты — и с ведущим
# пробелом, и без (SentencePiece часто ставит '_8' вместо '8'). Иначе судья молча
# суммирует не те токены и выдаёт правдоподобный мусор.
_SCORE_TOKENS: Dict[float, List[int]] = {}
for _d in range(11):
    _ids = []
    for _variant in (str(_d), " " + str(_d)):
        _toks = judge_tokenizer.encode(_variant, add_special_tokens=False)
        if len(_toks) == 1:
            _ids.append(int(_toks[0]))
    if _ids:
        _SCORE_TOKENS[float(_d)] = sorted(set(_ids))
log(f"judge score tokens: {len(_SCORE_TOKENS)} цифр имеют одно-токенные варианты")


@torch.no_grad()
def judge_hedge(question: str, answer: str) -> float:
    """Оценка увиливания в [0,10] = ожидание по цифрам (softmax по всему словарю)."""
    if not answer or not _SCORE_TOKENS:
        return float("nan")
    prompt = build_judge_prompt(question, answer)
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(JUDGE_DEVICE)
    out = judge_model(**inputs, use_cache=False)
    probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
    num, den = 0.0, 0.0
    for d, ids in _SCORE_TOKENS.items():
        p = float(sum(probs[i].item() for i in ids))
        num += d * p
        den += p
    # den = доля вероятности, реально ушедшая на цифры. Если крошечная — судья
    # не ответил числом, возвращаем NaN, чтобы не считать мусор.
    return float(num / den) if den > 1e-4 else float("nan")


def judge_selfcheck() -> None:
    """Быстрый тест: различает ли судья явный хедж и явную прямоту."""
    hedgy = ("Это очень сложный и неоднозначный вопрос. С одной стороны есть аргументы за, "
             "с другой — против. У каждой стороны своя правда, и всё зависит от обстоятельств.")
    direct = ("Да, безусловно, это нужно сделать. Позиция однозначна, и вот три конкретные "
              "причины, почему именно так, без всяких оговорок.")
    sh = judge_hedge("Тестовый вопрос?", hedgy)
    sd = judge_hedge("Тестовый вопрос?", direct)
    log(f"JUDGE self-check: hedgy={sh:.2f} (ждём ВЫСОКОЕ)  direct={sd:.2f} (ждём НИЗКОЕ)")
    if not (math.isfinite(sh) and math.isfinite(sd)) or sh <= sd:
        log("  ВНИМАНИЕ: судья не различает хедж и прямоту — проверь инструкцию/токены!")


# ── список слов (вторичная сверка) ──
def wordlist_lean_text(answer: str) -> float:
    a = (answer or "").lower()
    h = sum(a.count(w) for w in HEDGE_WORDS)
    d = sum(a.count(w) for w in DIRECT_WORDS)
    total = h + d
    return (h - d) / total if total > 0 else 0.0


# ── интервенция ──
@contextmanager
def intervention(vector_by_layer: np.ndarray, alpha: float, layer_indices: List[int]):
    layer_set = {int(i) for i in layer_indices if 1 <= int(i) <= N_LAYERS}
    handles = []

    def make_hook(li: int):
        vec_np = vector_by_layer[li]

        def hook(_module, _inputs, output):
            def modify(t):
                if not torch.is_tensor(t) or t.ndim != 3:
                    return t
                vec = torch.as_tensor(vec_np, device=t.device, dtype=t.dtype).view(1, 1, -1)
                out = t.clone()
                out[:, -1:, :] += float(alpha) * vec
                return out
            if torch.is_tensor(output):
                return modify(output)
            if isinstance(output, tuple) and output:
                items = list(output)
                items[0] = modify(items[0])
                return tuple(items)
            return output
        return hook

    try:
        for li, layer in enumerate(DECODER_LAYERS, start=1):
            if li in layer_set:
                handles.append(layer.register_forward_hook(make_hook(li)))
        yield
    finally:
        for h in handles:
            h.remove()


# ── генерация ответа ──
@torch.no_grad()
def generate_answer(prompt: str, vector_by_layer: Optional[np.ndarray] = None,
                    alpha: float = 0.0) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    ctx = (intervention(vector_by_layer, alpha, BAND)
           if vector_by_layer is not None else nullcontext())
    with ctx:
        out = model.generate(
            **inputs, max_new_tokens=ANSWER_MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def measure_behavior(question: str, full_prompt: str,
                     vector_by_layer: Optional[np.ndarray] = None, alpha: float = 0.0) -> dict:
    ans = generate_answer(full_prompt, vector_by_layer, alpha)
    return {"judge": judge_hedge(question, ans),
            "wordlist": wordlist_lean_text(ans), "answer": ans}


# ── случайные направления ──
def random_axis_like(reference_axis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r = np.zeros_like(reference_axis)
    for li in BAND:
        v = reference_axis[li]
        n = float(np.linalg.norm(v))
        g = rng.standard_normal(v.shape).astype(np.float32)
        g = g / (float(np.linalg.norm(g)) + 1e-12) * n
        r[li] = g
    return r


# ── утилиты ──
def corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    return float(pearsonr(x[ok], y[ok])[0]), float(spearmanr(x[ok], y[ok])[0])


def slope(d: dict) -> float:
    xs = np.array(sorted(d.keys()), dtype=np.float64)
    ys = np.array([d[a] for a in sorted(d.keys())], dtype=np.float64)
    if len(xs) < 2 or not np.all(np.isfinite(ys)):
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ЧАСТЬ A — наблюдение без толчка                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def run_part_a() -> dict:
    log("=" * 60)
    log("PART A: observational (no intervention)")
    log("=" * 60)

    conditions = [("neutral", NEUTRAL_TEXT)]
    conditions += [(f"target{i}", t) for i, t in enumerate(TARGET_TEXTS)]
    total = len(conditions) * len(QUESTIONS)
    rows = []

    for cond_name, ctx_text in conditions:
        for qi, q in enumerate(QUESTIONS):
            prompt = build_prompt(ctx_text, q)
            proj_o = projection_on_axis(prompt, AX[PRIMARY_AXIS])
            proj_c = projection_on_axis(prompt, AX[CONTENT_AXIS])
            beh = measure_behavior(q, prompt)
            rows.append({
                "condition": cond_name, "context": ctx_text, "qi": qi, "prompt": prompt,
                "proj_order": proj_o, "proj_content": proj_c,
                "hedge_judge": beh["judge"], "hedge_wordlist": beh["wordlist"],
                "answer": beh["answer"],
            })
            log(f"  [{len(rows)}/{total}] {cond_name} q{qi}: "
                f"proj_order={proj_o:.3f}  judge={beh['judge']:.2f}  wl={beh['wordlist']:+.2f}")
            # санити-лог: первые 2 ответа целиком, чтобы глазами проверить судью
            if len(rows) <= 2:
                log(f"      ОТВЕТ: {beh['answer'][:200]!r}")

    judge_arr = np.array([r["hedge_judge"] for r in rows], dtype=np.float64)
    wl_arr    = np.array([r["hedge_wordlist"] for r in rows], dtype=np.float64)
    po_arr    = np.array([r["proj_order"] for r in rows], dtype=np.float64)
    pc_arr    = np.array([r["proj_content"] for r in rows], dtype=np.float64)

    r_order,   rho_order   = corr(po_arr, judge_arr)
    r_content, rho_content = corr(pc_arr, judge_arr)
    r_order_wl, _          = corr(po_arr, wl_arr)

    log(f"  random baseline (N={N_RANDOM}) — все промпты уже в кэше, forward'ов нет ...")
    rng = np.random.default_rng(SEED)
    rnd_rs = []
    for _ in tqdm(range(N_RANDOM), desc="random baseline A"):
        rax = random_axis_like(AX[PRIMARY_AXIS], rng)
        pr = np.array([projection_on_axis(r["prompt"], rax) for r in rows], dtype=np.float64)
        rr, _ = corr(pr, judge_arr)
        if np.isfinite(rr):
            rnd_rs.append(abs(rr))
    rnd_p95 = float(np.percentile(rnd_rs, 95)) if rnd_rs else float("nan")

    summary = {
        "pearson_order_vs_judge": r_order, "spearman_order_vs_judge": rho_order,
        "pearson_content_vs_judge": r_content, "spearman_content_vs_judge": rho_content,
        "pearson_order_vs_wordlist_secondary": r_order_wl,
        "random_abs_pearson_p95": rnd_p95,
        "order_beats_random": (bool(abs(r_order) > rnd_p95)
                               if math.isfinite(r_order) and math.isfinite(rnd_p95) else None),
        "order_beats_content": (bool(abs(r_order) >= abs(r_content))
                                if math.isfinite(r_order) and math.isfinite(r_content) else None),
        "n_points": len(rows),
    }

    rows_save = [{k: v for k, v in r.items() if k != "prompt"} for r in rows]
    json.dump({"rows": rows_save, "summary": summary},
              open(OUTPUT_DIR / "part_a.json", "w"), ensure_ascii=False, indent=2)
    log(f"PART A summary: {json.dumps(summary, ensure_ascii=False)}")
    return summary


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ЧАСТЬ B — лёгкий толчок                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def run_part_b() -> dict:
    log("=" * 60)
    log("PART B: causal (inject +/- alpha * axis)")
    log("=" * 60)

    qprompts = [(q, build_prompt(NEUTRAL_TEXT, q)) for q in QUESTIONS]
    rng = np.random.default_rng(SEED + 1)
    n_rand_b = min(N_RANDOM, 10)
    rand_axes = [random_axis_like(AX[PRIMARY_AXIS], rng) for _ in range(n_rand_b)]

    def gap_for_axis(vec_by_layer: np.ndarray, alpha: float) -> float:
        gaps = []
        for q, p in qprompts:
            lp = measure_behavior(q, p, vec_by_layer, +alpha)["judge"]
            lm = measure_behavior(q, p, vec_by_layer, -alpha)["judge"]
            if np.isfinite(lp) and np.isfinite(lm):
                gaps.append(lp - lm)
        return float(np.mean(gaps)) if gaps else float("nan")

    results: Dict[str, Dict] = {"order": {}, "content": {}}
    n_main = len(ALPHAS) * 2
    step = 0
    for alpha in ALPHAS:
        log(f"  alpha={alpha} ...")
        results["order"][alpha]   = gap_for_axis(AX[PRIMARY_AXIS], alpha)
        results["content"][alpha] = gap_for_axis(AX[CONTENT_AXIS], alpha)
        step += 2
        log(f"  [{step}/{n_main}] alpha={alpha}: "
            f"order_gap={results['order'][alpha]:+.4f}  content_gap={results['content'][alpha]:+.4f}")

    a_max = max(ALPHAS)
    log(f"  random baseline (n={n_rand_b}) at alpha={a_max} ...")
    rnd_gaps = []
    for rax in tqdm(rand_axes, desc="random baseline B"):
        g = abs(gap_for_axis(rax, a_max))
        if np.isfinite(g):
            rnd_gaps.append(g)
    rnd_p95 = float(np.percentile(rnd_gaps, 95)) if rnd_gaps else float("nan")

    order_gap_max = results["order"][a_max]
    summary = {
        "order_gaps_by_alpha": results["order"], "content_gaps_by_alpha": results["content"],
        "order_slope": slope(results["order"]), "content_slope": slope(results["content"]),
        "random_abs_gap_p95_at_alpha_max": rnd_p95,
        "order_beats_random": (bool(abs(order_gap_max) > rnd_p95) if math.isfinite(rnd_p95) else None),
        "order_monotonic_positive": bool(slope(results["order"]) > 0),
        "alpha_max": a_max,
    }
    json.dump(summary, open(OUTPUT_DIR / "part_b.json", "w"), ensure_ascii=False, indent=2)
    log(f"PART B summary: {json.dumps(summary, ensure_ascii=False)}")
    return summary


# ── main ──
if __name__ == "__main__":
    judge_selfcheck()
    a = run_part_a()
    b = run_part_b()
    verdict = {
        "part_a_order_predicts_behavior":       a.get("order_beats_random"),
        "part_a_order_not_weaker_than_content": a.get("order_beats_content"),
        "part_b_order_steers_behavior":         b.get("order_beats_random"),
        "part_b_dose_response_ok":              b.get("order_monotonic_positive"),
    }
    json.dump(verdict, open(OUTPUT_DIR / "verdict.json", "w"), ensure_ascii=False, indent=2)
    log(f"VERDICT: {json.dumps(verdict, ensure_ascii=False)}")
    print("\n==== ИТОГ ====")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
