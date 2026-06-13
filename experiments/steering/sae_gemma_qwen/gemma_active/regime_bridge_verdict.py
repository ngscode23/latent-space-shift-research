#!/usr/bin/env python3
# ============================================================
# REGIME BRIDGE — VERDICT ANALYZER
# Достраивает вывод поверх codex-скрипта (regime_axis_grade_bridge_causal_audit.py).
# Не пересобирает активации: работает с regime_bridge_causal_generation_*.csv.
#
# Закрывает дыры исходника:
#   1) actual - random по ПОВЕДЕНЧЕСКОЙ метрике, с bootstrap-CI (свой нуль у причинного теста).
#   2) "ворота цели": на сколько control+regime закрывает разрыв до target-уровня
#      (и target-regime — до control). Ловит "ушёл от себя, но не к цели".
#   3) directness нормирован на длину и очищен от строк со сменой языка.
#   4) язык-агностичная semantic-близость к target/control базам (LaBSE), которой нет в исходнике.
#
# Запуск:  python regime_bridge_verdict.py regime_bridge_causal_generation_<TAG>.csv
# ============================================================

import sys, os, re, argparse
import numpy as np
import pandas as pd

ACTUAL = "actual"
RANDOM = "random_same_norm_causal"
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")


# ---------- надёжные поведенческие метрики (на случай старого CSV без них) ----------

def directness_norm(row):
    """directness на 100 слов; None если строка со сменой языка (маркеры слепнут)."""
    if int(row.get("script_switch_flag", 0)) == 1:
        return np.nan
    w = max(int(row.get("output_words", 0)) or len(_WORD.findall(str(row.get("output_text", "")))), 1)
    dp = row.get("directness_proxy", np.nan)
    return float(dp) / w * 100.0 if pd.notna(dp) else np.nan


def _side_of(direction, base_side):
    if isinstance(base_side, str) and base_side in ("control", "target"):
        return base_side
    d = str(direction)
    return "control" if d.startswith("control") else "target"


def _is_random(axis_source):
    return str(axis_source) == RANDOM


def bootstrap_ci(vals, n=2000, lo=2.5, hi=97.5, seed=0):
    vals = np.asarray([v for v in vals if v == v], dtype=float)
    if len(vals) < 2:
        return (float("nan"), float("nan"), float(np.mean(vals)) if len(vals) else float("nan"))
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi)), float(np.mean(vals))


def nearest_alpha(target_a, available):
    available = sorted(set(available))
    return min(available, key=lambda a: abs(a - target_a)) if available else None


def analyze(df, encoder=None):
    df = df.copy()
    if "direction" not in df.columns or "alpha_mult" not in df.columns:
        raise ValueError("Это не causal CSV (нет direction/alpha_mult).")
    df["alpha_mult"] = df["alpha_mult"].astype(float)
    df["side"] = [_side_of(d, b) for d, b in zip(df["direction"], df.get("base_side", df["direction"]))]
    df["is_random"] = df["axis_source"].map(_is_random)
    df["dir_norm"] = df.apply(directness_norm, axis=1)

    n_switch = int((df.get("script_switch_flag", pd.Series(dtype=int)) == 1).sum())
    print(f"строк всего: {len(df)} | со сменой языка (исключены из directness): {n_switch}")

    # --- baselines (actual, alpha=0), усредняем по вариантам ---
    base = df[(~df.is_random) & (df.alpha_mult == 0.0)]
    ctrl_base = base[base.side == "control"]["dir_norm"].mean()
    tgt_base = base[base.side == "target"]["dir_norm"].mean()
    gap = tgt_base - ctrl_base
    print(f"\nbaseline directness/100w:  control={ctrl_base:+.3f}  target={tgt_base:+.3f}  gap(target-control)={gap:+.3f}")
    if abs(gap) < 1e-9:
        print("  [!] gap≈0: цели совпадают по directness — 'ворота цели' неинформативны, смотрите semantic ниже.")

    variants = [v for v in df[~df.is_random]["axis_variant"].unique()]
    rand_alphas = sorted(df[df.is_random]["alpha_mult"].unique())

    rows = []
    for variant in variants:
        for side in ("control", "target"):
            sub_a = df[(~df.is_random) & (df.axis_variant == variant) & (df.side == side)]
            b = ctrl_base if side == "control" else tgt_base
            for a in sorted(x for x in sub_a["alpha_mult"].unique() if x != 0.0):
                act = sub_a[sub_a.alpha_mult == a]["dir_norm"]
                shift_act = act.mean() - b
                # random на ближайшей доступной альфе той же стороны
                ra = nearest_alpha(a, rand_alphas)
                rnd = df[(df.is_random) & (df.side == side) & (df.alpha_mult == ra)]["dir_norm"] if ra is not None else pd.Series(dtype=float)
                shift_rnd = (rnd.mean() - b) if len(rnd) else np.nan
                effect = shift_act - shift_rnd  # сдвиг сверх случайного толчка той же нормы
                lo, hi, mean_eff = bootstrap_ci(act.dropna().values - (rnd.mean() if len(rnd) else 0.0))
                # ворота цели: какую долю разрыва до ДРУГОЙ стороны закрыл actual-сдвиг
                if side == "control" and abs(gap) > 1e-9:
                    gap_closed = shift_act / gap
                elif side == "target" and abs(gap) > 1e-9:
                    gap_closed = -shift_act / gap  # target движется к control => к -gap
                else:
                    gap_closed = np.nan
                rows.append(dict(
                    variant=variant, side=side, alpha=a,
                    shift_actual=round(shift_act, 3),
                    shift_random=round(shift_rnd, 3) if shift_rnd == shift_rnd else np.nan,
                    effect_vs_random=round(effect, 3) if effect == effect else np.nan,
                    eff_CI=f"[{lo:+.2f},{hi:+.2f}]",
                    gap_closed=round(gap_closed, 3) if gap_closed == gap_closed else np.nan,
                    n_actual=int(act.notna().sum()), n_random=int(rnd.notna().sum()) if len(rnd) else 0,
                ))
    out = pd.DataFrame(rows)
    print("\n=== ПОВЕДЕНЧЕСКИЙ ЭФФЕКТ (directness/100w, строки без смены языка) ===")
    print("effect_vs_random = (actual-сдвиг) − (случайный-сдвиг той же нормы); CI не должен включать 0.")
    print("gap_closed: 0=не сдвинулся к цели, 1=дошёл до уровня цели, <0=ушёл В ДРУГУЮ сторону.")
    if len(out):
        print(out.to_string(index=False))
    else:
        print("  нет строк для анализа.")

    # --- опциональная semantic-близость (язык-агностичная) ---
    if encoder is not None and encoder.ok:
        print("\n=== SEMANTIC к target/control базам (LaBSE; робастно к языку) ===")
        _semantic_block(df, encoder)
    else:
        print("\n[semantic] пропущено (нет sentence-transformers). Поставьте для язык-агностичной 'ворот цели':")
        print("           pip install sentence-transformers")
    return out


def _semantic_block(df, enc):
    # представительные базы по задаче: берём actual alpha=0
    base = df[(~df.is_random) & (df.alpha_mult == 0.0)]
    tgt_base_by_task, ctrl_base_by_task = {}, {}
    for _, r in base.iterrows():
        key = r.get("task_id", 0)
        (tgt_base_by_task if r["side"] == "target" else ctrl_base_by_task)[key] = str(r.get("baseline_output", ""))
    texts = set()
    for d in (tgt_base_by_task, ctrl_base_by_task):
        texts.update(d.values())
    texts.update(str(t) for t in df["output_text"].dropna().tolist())
    enc.encode_many([t for t in texts if t.strip()])

    rows = []
    for variant in df[~df.is_random]["axis_variant"].unique():
        for side in ("control", "target"):
            sub = df[(~df.is_random) & (df.axis_variant == variant) & (df.side == side)]
            for a in sorted(sub["alpha_mult"].unique()):
                grp = sub[sub.alpha_mult == a]
                sims_tgt, sims_ctrl = [], []
                for _, r in grp.iterrows():
                    o = str(r.get("output_text", "")); tk = r.get("task_id", 0)
                    tb, cb = tgt_base_by_task.get(tk, ""), ctrl_base_by_task.get(tk, "")
                    if tb: sims_tgt.append(enc.cos(o, tb))
                    if cb: sims_ctrl.append(enc.cos(o, cb))
                rows.append(dict(variant=variant, side=side, alpha=a,
                                 sim_to_target=round(np.nanmean(sims_tgt), 3) if sims_tgt else np.nan,
                                 sim_to_control=round(np.nanmean(sims_ctrl), 3) if sims_ctrl else np.nan,
                                 n=len(grp)))
    sem = pd.DataFrame(rows)
    print("Хотим для control+regime: sim_to_target ↑, sim_to_control ↓ по мере роста |alpha|.")
    print(sem.to_string(index=False))


class Encoder:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        self.ok = False; self._c = {}
        try:
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(model_name); self.ok = True
        except Exception as e:
            print(f"[semantic] энкодер недоступен: {type(e).__name__}")

    def encode_many(self, texts):
        todo = [t for t in {str(x) for x in texts} if t.strip() and t not in self._c]
        if todo and self.ok:
            import numpy as _np
            v = self.m.encode(todo, normalize_embeddings=True, show_progress_bar=False)
            for t, vec in zip(todo, _np.asarray(v, dtype=_np.float32)):
                self._c[t] = vec

    def cos(self, a, b):
        a, b = str(a), str(b)
        if not self.ok or a not in self._c or b not in self._c:
            return np.nan
        return float(np.dot(self._c[a], self._c[b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("causal_csv")
    ap.add_argument("--no-semantic", action="store_true")
    args, _ = ap.parse_known_args()
    df = pd.read_csv(args.causal_csv)
    enc = None if args.no_semantic else Encoder()
    out = analyze(df, encoder=enc)
    out_path = os.path.splitext(args.causal_csv)[0] + "_VERDICT.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[ok] verdict-таблица -> {out_path}")
    print("\nЧитать: сильный причинный сигнал = effect_vs_random с CI вне нуля И gap_closed заметно >0")
    print("        (а если есть semantic — sim_to_target растёт). Иначе ось двигает 'не туда' или как случайная.")


if __name__ == "__main__":
    main()
