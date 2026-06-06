# ==============================================================================
# SCALE CALIBRATION
# Хук добавляет scale * W_dec[f] на ВСЕ позиции residual stream.
# Полезный сигнал = норма добавки как доля нормы residual'а.
# ==============================================================================

# ------------------------------------------------------------------------------
# FEATURE 41/13686
#   ||W_dec[f]||                  = 1.0000
#   ||residual|| @L41 median     = 127086.9   (mean 130505.7)
#   нативная активация (active):    n=58  доля=0.027  median=2561.3  mean=2779.4  p95=3913.4  max=4688.8
#   нативный вклад фичи (median_act*||W_dec||) = 2561.3  = 2.02% нормы residual'а
#   --- ТВОИ СТАРЫЕ scale ---
#     scale +1.5 -> добавка=1.50  = 0.001% residual'а  = 0.0006x нативного срабатывания
#     scale +3.0 -> добавка=3.00  = 0.002% residual'а  = 0.0012x нативного срабатывания
#   --- РЕКОМЕНДАЦИЯ (scale под долю нормы residual'а) ---
#       5% residual'а -> scale =     6350.0   (= 2.48x нативного)
#      10% residual'а -> scale =    12700.0   (= 4.96x нативного)
#      20% residual'а -> scale =    25400.0   (= 9.92x нативного)
#      50% residual'а -> scale =    63500.0   (= 24.81x нативного)
#     100% residual'а -> scale =   127000.0   (= 49.62x нативного)
#   --- справочно (scale = кратность нативного срабатывания) ---
#     1x нативного -> scale =     2560.0   (= 2.0% residual'а)
#     2x нативного -> scale =     5120.0   (= 4.0% residual'а)
#     4x нативного -> scale =    10200.0   (= 8.1% residual'а)
#     8x нативного -> scale =    20500.0   (= 16.1% residual'а)
#   >>> STEERING_SCALES (готово к вставке): [-63500.0, -25400.0, -12700.0, -6350.0, 0.0, 6350.0, 12700.0, 25400.0, 63500.0]

# ------------------------------------------------------------------------------
# FEATURE 41/208
#   ||W_dec[f]||                  = 1.0000
#   ||residual|| @L41 median     = 127086.9   (mean 130505.7)
#   нативная активация (active):    n=1025  доля=0.479  median=9579.7  mean=11134.8  p95=22837.4  max=30323.3
#   нативный вклад фичи (median_act*||W_dec||) = 9579.7  = 7.54% нормы residual'а
#   --- ТВОИ СТАРЫЕ scale ---
#     scale +1.5 -> добавка=1.50  = 0.001% residual'а  = 0.0002x нативного срабатывания
#     scale +3.0 -> добавка=3.00  = 0.002% residual'а  = 0.0003x нативного срабатывания
#   --- РЕКОМЕНДАЦИЯ (scale под долю нормы residual'а) ---
#       5% residual'а -> scale =     6350.0   (= 0.66x нативного)
#      10% residual'а -> scale =    12700.0   (= 1.33x нативного)
#      20% residual'а -> scale =    25400.0   (= 2.65x нативного)
#      50% residual'а -> scale =    63500.0   (= 6.63x нативного)
#     100% residual'а -> scale =   127000.0   (= 13.27x нативного)
#   --- справочно (scale = кратность нативного срабатывания) ---
#     1x нативного -> scale =     9580.0   (= 7.5% residual'а)
#     2x нативного -> scale =    19200.0   (= 15.1% residual'а)
#     4x нативного -> scale =    38300.0   (= 30.2% residual'а)
#     8x нативного -> scale =    76600.0   (= 60.3% residual'а)
#   >>> STEERING_SCALES (готово к вставке): [-63500.0, -25400.0, -12700.0, -6350.0, 0.0, 6350.0, 12700.0, 25400.0, 63500.0]

# ------------------------------------------------------------------------------
# FEATURE 41/207
#   ||W_dec[f]||                  = 1.0000
#   ||residual|| @L41 median     = 127086.9   (mean 130505.7)
#   нативная активация (active):    n=495  доля=0.231  median=2899.6  mean=3900.9  p95=5236.4  max=108523.8
#   нативный вклад фичи (median_act*||W_dec||) = 2899.6  = 2.28% нормы residual'а
#   --- ТВОИ СТАРЫЕ scale -
#     scale +1.5 -> добавка=1.50  = 0.001% residual'а  = 0.0005x нативного срабатывания
#     scale +3.0 -> добавка=3.00  = 0.002% residual'а  = 0.0010x нативного срабатывания
#   --- РЕКОМЕНДАЦИЯ (scale под долю нормы residual'а) ---
#       5% residual'а -> scale =     6350.0   (= 2.19x нативного)
#      10% residual'а -> scale =    12700.0   (= 4.38x нативного)
#      20% residual'а -> scale =    25400.0   (= 8.77x нативного)
#      50% residual'а -> scale =    63500.0   (= 21.91x нативного)
#     100% residual'а -> scale =   127000.0   (= 43.83x нативного)
#   --- справочно (scale = кратность нативного срабатывания) ---
#     1x нативного -> scale =     2900.0   (= 2.3% residual'а)
#     2x нативного -> scale =     5800.0   (= 4.6% residual'а)
#     4x нативного -> scale =    11600.0   (= 9.1% residual'а)
#     8x нативного -> scale =    23200.0   (= 18.3% residual'а)
#   >>> STEERING_SCALES (готово к вставке): [-63500.0, -25400.0, -12700.0, -6350.0, 0.0, 6350.0, 12700.0, 25400.0, 63500.0]

# Сохранено: sae_scale_calibration.csv

# ==============================================================================
# RECOMMENDED_SCALES_BY_FEATURE (можно копировать в steering-скрипт):
# ==============================================================================
# RECOMMENDED_SCALES_BY_FEATURE = {
#     (41, 13686): [-63500.0, -25400.0, -12700.0, -6350.0, 0.0, 6350.0, 12700.0, 25400.0, 63500.0],
#     (41, 208): [-63500.0, -25400.0, -12700.0, -6350.0, 0.0, 6350.0, 12700.0, 25400.0, 63500.0],
#     (41, 207): [-63500.0, -25400.0, -12700.0, -6350.0, 0.0, 6350.0, 12700.0, 25400.0, 63500.0],
# }

# ==============================================================================
# KL CHECK: next-token KL(base||patched) на последней позиции промпта.
# Цель — увидеть, что на рекомендованном scale KL уже НЕ шум (>~0.01),
# и в идеале top-токен начинает меняться.
# ==============================================================================

# FEATURE 41/13686
#   OLD scale=3.0  (scale=      3.00) -> KL=0.00001  logit_l2=  10.88  [top same]
#   10% resid      (scale=  12708.69) -> KL=0.14368  logit_l2= 573.59  [top same]
#   50% resid      (scale=  63543.44) -> KL=10.47911  logit_l2=2123.92  [TOP CHANGED]

# FEATURE 41/208
#   OLD scale=3.0  (scale=      3.00) -> KL=0.00001  logit_l2=  10.93  [top same]
#   10% resid      (scale=  12708.69) -> KL=0.01097  logit_l2= 283.90  [top same]
#   50% resid      (scale=  63543.44) -> KL=2.27440  logit_l2=1429.28  [TOP CHANGED]

# FEATURE 41/207
#   OLD scale=3.0  (scale=      3.00) -> KL=0.00001  logit_l2=  11.09  [top same]
#   10% resid      (scale=  12708.69) -> KL=0.00240  logit_l2= 274.58  [top same]
#   50% resid      (scale=  63543.44) -> KL=0.17687  logit_l2=2498.64  [top same]

# Сохранено: sae_scale_calibration_kl_check.csv

# ГОТОВО. Бери STEERING_SCALES из RECOMMENDED_SCALES_BY_FEATURE и перезапусти
# sae_steering_with_kl_full.py — teacher-forced KL должен ожить.

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# ====================== НАСТРОЙКИ ======================
STEERING_FEATURES = [
    (41, 13686),  # semantic_marker_target_only
    (41, 208),    # contrastive_rhetorical_marker (?)
    (41, 207),    # strong_dirty_causal_driver
]

# Доли нормы residual stream, под которые подбираем scale.
# 0.05–0.5 — рабочий коридор: ниже обычно ничего не двигается,
# выше ~1.0 генерация почти гарантированно разваливается в мусор.
RESID_FRACTIONS = [0.05, 0.10, 0.20, 0.50, 1.00]

OLD_SCALES = [1.5, 3.0]          # что ты гонял раньше — для сравнения
ACTIVE_THRESHOLD = 1e-6          # активной считаем позицию с act > этого

# Какие доли взять в итоговый paste-ready симметричный грид
PASTE_FRACTIONS = [0.05, 0.10, 0.20, 0.50]

# Проверка KL: на каких долях residual'а прогнать next-token KL
RUN_KL_CHECK = True
KL_CHECK_FRACTIONS = [0.10, 0.50]
KL_CHECK_TASK = (
    "Сожми текст до одного беспощадного вывода.\n"
    "Одна фраза. Без оговорок."
)

BASE_TEXT = prompts_target[0]


# ====================== ВСПОМОГАТЕЛЬНОЕ ======================
def get_resid_and_latent(text, real_layer):
    """Возвращает (resid[1,seq,d_model] float, latent[1,seq,n_feat] float) на hook_resid_post."""
    sae = saes[real_layer]
    hook_name = f"blocks.{real_layer}.hook_resid_post"
    with torch.no_grad():
        tokens = model.to_tokens(text, prepend_bos=True)
        if hasattr(model.cfg, "n_ctx") and tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, -model.cfg.n_ctx:]
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name].float()        # [1, seq, d_model]
        latent = sae.encode(resid)              # [1, seq, n_feat]
    return resid, latent


def round_sig(x, sig=3):
    if x == 0 or not np.isfinite(x):
        return 0.0
    return float(round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1)))


def build_analysis_prompt(base_text, task):
    return (
        "Ты анализируешь один и тот же текст.\n\n"
        "=== ТЕКСТ ДЛЯ АНАЛИЗА ===\n"
        f"{base_text}\n\n"
        "=== ЗАДАНИЕ ===\n"
        f"{task.strip()}\n\n"
        "=== ОТВЕТ ===\n"
    )


def steer_hook_factory(real_layer, feature_index, scale):
    sae = saes[real_layer]
    f_idx = int(feature_index)

    def hook(activation, hook, _sae=sae, _f=f_idx, _s=float(scale)):
        orig_dtype = activation.dtype
        dec_vec = _sae.W_dec[_f].to(device=activation.device, dtype=orig_dtype)
        return activation + _s * dec_vec

    return hook


def next_token_kl(prompt, real_layer, feature_index, scale):
    """KL(p_base || p_patched) на последней позиции промпта + смена top-токена."""
    hook_name = f"blocks.{real_layer}.hook_resid_post"
    with torch.no_grad():
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if hasattr(model.cfg, "n_ctx") and tokens.shape[1] > model.cfg.n_ctx:
            tokens = tokens[:, -model.cfg.n_ctx:]

        base_logits = model(tokens)[:, -1, :].float()
        hook = steer_hook_factory(real_layer, feature_index, scale)
        with model.hooks(fwd_hooks=[(hook_name, hook)]):
            patched_logits = model(tokens)[:, -1, :].float()

        log_pb = F.log_softmax(base_logits, dim=-1)
        log_pp = F.log_softmax(patched_logits, dim=-1)
        pb = log_pb.exp()
        kl = (pb * (log_pb - log_pp)).sum(-1).item()
        logit_l2 = (patched_logits - base_logits).norm().item()
        top_base = int(base_logits.argmax(-1).item())
        top_patched = int(patched_logits.argmax(-1).item())
    return kl, logit_l2, top_base, top_patched


# ====================== ОСНОВНОЙ ЦИКЛ ======================
rows = []
recommended = {}

print("=" * 78)
print("SCALE CALIBRATION")
print("Хук добавляет scale * W_dec[f] на ВСЕ позиции residual stream.")
print("Полезный сигнал = норма добавки как доля нормы residual'а.")
print("=" * 78)

for real_layer, feature_index in STEERING_FEATURES:
    sae = saes[real_layer]

    # 1) норма decoder-направления
    w_dec_norm = sae.W_dec[int(feature_index)].float().norm().item()

    # 2) residual + 3) нативная активация на BASE_TEXT
    resid, latent = get_resid_and_latent(BASE_TEXT, real_layer)
    resid_norms = resid[0].norm(dim=-1).cpu().numpy()       # [seq]
    med_resid = float(np.median(resid_norms))
    mean_resid = float(resid_norms.mean())

    act = latent[0, :, int(feature_index)].cpu().numpy()    # [seq]
    act_active = act[act > ACTIVE_THRESHOLD]
    if act_active.size > 0:
        med_act = float(np.median(act_active))
        mean_act = float(act_active.mean())
        p95_act = float(np.percentile(act_active, 95))
        max_act = float(act_active.max())
        frac_active = float((act > ACTIVE_THRESHOLD).mean())
    else:
        med_act = mean_act = p95_act = max_act = 0.0
        frac_active = 0.0

    # норма того, что фича добавляет САМА при типичном срабатывании
    native_contrib_norm = med_act * w_dec_norm
    native_pct_of_resid = 100.0 * native_contrib_norm / med_resid if med_resid else float("nan")

    print(f"\n{'-'*78}")
    print(f"FEATURE {real_layer}/{feature_index}")
    print(f"  ||W_dec[f]||                  = {w_dec_norm:.4f}")
    print(f"  ||residual|| @L{real_layer} median     = {med_resid:.1f}   (mean {mean_resid:.1f})")
    print(f"  нативная активация (active):    n={act_active.size}  доля={frac_active:.3f}  "
          f"median={med_act:.1f}  mean={mean_act:.1f}  p95={p95_act:.1f}  max={max_act:.1f}")
    print(f"  нативный вклад фичи (median_act*||W_dec||) = {native_contrib_norm:.1f}  "
          f"= {native_pct_of_resid:.2f}% нормы residual'а")

    # 4) чем был твой старый scale
    print(f"  --- ТВОИ СТАРЫЕ scale ---")
    for s in OLD_SCALES:
        added = s * w_dec_norm
        pct = 100.0 * added / med_resid if med_resid else float("nan")
        x_native = added / native_contrib_norm if native_contrib_norm else float("nan")
        print(f"    scale {s:+.1f} -> добавка={added:.2f}  = {pct:.3f}% residual'а  = {x_native:.4f}x нативного срабатывания")

    # 5) рекомендованные scale по доле residual'а
    print(f"  --- РЕКОМЕНДАЦИЯ (scale под долю нормы residual'а) ---")
    frac_to_scale = {}
    for frac in RESID_FRACTIONS:
        s = frac * med_resid / w_dec_norm if w_dec_norm else float("nan")
        frac_to_scale[frac] = s
        x_native = (s * w_dec_norm) / native_contrib_norm if native_contrib_norm else float("nan")
        print(f"    {int(frac*100):3d}% residual'а -> scale = {round_sig(s):>10}   (= {x_native:.2f}x нативного)")

    # справочно: scale под кратность нативного срабатывания
    if native_contrib_norm:
        print(f"  --- справочно (scale = кратность нативного срабатывания) ---")
        for mult in [1, 2, 4, 8]:
            s = mult * med_act
            pct = 100.0 * (s * w_dec_norm) / med_resid if med_resid else float("nan")
            print(f"    {mult}x нативного -> scale = {round_sig(s):>10}   (= {pct:.1f}% residual'а)")

    # paste-ready симметричный грид
    pos = [round_sig(frac_to_scale[f]) for f in PASTE_FRACTIONS]
    grid = sorted(set([-p for p in pos] + [0.0] + pos))
    recommended[(real_layer, feature_index)] = grid
    print(f"  >>> STEERING_SCALES (готово к вставке): {grid}")

    rows.append({
        "real_layer": real_layer,
        "feature_index": feature_index,
        "w_dec_norm": w_dec_norm,
        "resid_norm_median": med_resid,
        "resid_norm_mean": mean_resid,
        "native_act_median": med_act,
        "native_act_mean": mean_act,
        "native_act_p95": p95_act,
        "native_act_max": max_act,
        "frac_active": frac_active,
        "native_contrib_norm": native_contrib_norm,
        "native_pct_of_resid": native_pct_of_resid,
        **{f"scale_for_{int(f*100)}pct_resid": frac_to_scale[f] for f in RESID_FRACTIONS},
        "recommended_grid": str(grid),
    })

# ====================== СОХРАНЕНИЕ ТАБЛИЦЫ ======================
calib_df = pd.DataFrame(rows)
calib_df.to_csv("sae_scale_calibration.csv", index=False)
print(f"\nСохранено: sae_scale_calibration.csv")

print("\n" + "=" * 78)
print("RECOMMENDED_SCALES_BY_FEATURE (можно копировать в steering-скрипт):")
print("=" * 78)
print("RECOMMENDED_SCALES_BY_FEATURE = {")
for k, v in recommended.items():
    print(f"    {k}: {v},")
print("}")


# ====================== ОПЦИОНАЛЬНО: ПРОВЕРКА KL ======================
if RUN_KL_CHECK:
    print("\n" + "=" * 78)
    print("KL CHECK: next-token KL(base||patched) на последней позиции промпта.")
    print("Цель — увидеть, что на рекомендованном scale KL уже НЕ шум (>~0.01),")
    print("и в идеале top-токен начинает меняться.")
    print("=" * 78)

    prompt = build_analysis_prompt(BASE_TEXT, KL_CHECK_TASK)
    kl_rows = []

    for real_layer, feature_index in STEERING_FEATURES:
        sae = saes[real_layer]
        w_dec_norm = sae.W_dec[int(feature_index)].float().norm().item()
        resid, _ = get_resid_and_latent(BASE_TEXT, real_layer)
        med_resid = float(np.median(resid[0].norm(dim=-1).cpu().numpy()))

        print(f"\nFEATURE {real_layer}/{feature_index}")
        # baseline sanity: scale=0 должен дать KL=0
        for label, scale in (
            [("OLD scale=3.0", 3.0)]
            + [(f"{int(f*100)}% resid", f * med_resid / w_dec_norm) for f in KL_CHECK_FRACTIONS]
        ):
            kl, l2, tb, tp = next_token_kl(prompt, real_layer, feature_index, scale)
            changed = "TOP CHANGED" if tb != tp else "top same"
            print(f"  {label:14s} (scale={scale:10.2f}) -> KL={kl:.5f}  logit_l2={l2:7.2f}  [{changed}]")
            kl_rows.append({
                "real_layer": real_layer, "feature_index": feature_index,
                "label": label, "scale": scale, "kl_base_to_patched": kl,
                "logit_l2": l2, "top_token_changed": int(tb != tp),
            })

    kl_check_df = pd.DataFrame(kl_rows)
    kl_check_df.to_csv("sae_scale_calibration_kl_check.csv", index=False)
    print(f"\nСохранено: sae_scale_calibration_kl_check.csv")

print("\nГОТОВО. Бери STEERING_SCALES из RECOMMENDED_SCALES_BY_FEATURE и перезапусти")
print("sae_steering_with_kl_full.py — teacher-forced KL должен ожить.")