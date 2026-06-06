# =============================================
# FEATURE STEERING GENERATION TEST v2
# sampling + новые таски + model.hooks()
# =============================================

import torch
import pandas as pd
from tqdm import tqdm

# ====================== НАСТРОЙКИ ======================
STEERING_FEATURES = [
    (41, 13686),  # главный semantic marker
    (41, 208),    # contrastive / rhetorical candidate
    (41, 207),    # strong but dirty causal candidate
]

STEERING_SCALES = [-3.0, -1.5, 0.0, 1.5, 3.0]

N_SAMPLES    = 5      # сэмплов на каждую комбинацию
DO_SAMPLE    = True
TEMPERATURE  = 0.8
MAX_NEW_TOKENS = 220

# BASE_TEXT берётся из prompts_target[0] — убедись что он определён выше в ноутбуке
BASE_TEXT = prompts_target[0]

# ====================== ТАСКИ ======================
# Убрали слишком узкое "одна фраза" как единственное задание,
# добавили таски с бо́льшей свободой — steering проявляется лучше
TEST_TASKS = [
    """
Сожми текст до одного беспощадного вывода.
Одна фраза. Без оговорок.
""",

    """
Напиши 6 разных формулировок главного диагноза этого текста.
Каждая формулировка должна быть короткой, жёсткой и аналитической.
""",

    """
Продолжи мысль текста на 150 слов.
Сохрани холодный диагностический режим.
""",

    """
Перепиши главный тезис текста в ещё более сухой и административно-жёсткой форме.
""",

    """
Выдели механизм слабости, описанный в тексте, и сформулируй его как технический дефект.
""",
]

# ====================== ПРОМПТ-BUILDER ======================
def build_analysis_prompt(base_text, task):
    return (
        "Ты анализируешь один и тот же текст.\n\n"
        "=== ТЕКСТ ДЛЯ АНАЛИЗА ===\n"
        f"{base_text}\n\n"
        "=== ЗАДАНИЕ ===\n"
        f"{task.strip()}\n\n"
        "=== ОТВЕТ ===\n"
    )

# ====================== STEERING HOOK ======================
def steer_sae_feature_all_positions(activation, hook, real_layer, feature_index, scale=1.0):
    """Добавляет decoder direction выбранной SAE-фичи на все позиции residual stream."""
    sae = saes[real_layer]
    orig_dtype = activation.dtype
    act_float  = activation.float()

    f_idx = int(feature_index)
    if f_idx < 0 or f_idx >= sae.W_dec.shape[0]:
        return activation

    with torch.no_grad():
        dec_vec = sae.W_dec[f_idx].to(device=act_float.device, dtype=act_float.dtype)
        patched  = act_float + scale * dec_vec

    return patched.to(dtype=orig_dtype)

# ====================== ГЕНЕРАЦИЯ ======================
def generate_safely(prompt, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE, temperature=TEMPERATURE):
    """Совместимость с разными версиями TransformerLens generate()."""
    try:
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            verbose=False,
        )
    except TypeError:
        # старые версии не принимают do_sample / temperature
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            verbose=False,
        )


def generate_with_feature_steering(prompt, real_layer, feature_index, scale,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    do_sample=DO_SAMPLE, temperature=TEMPERATURE):
    hook_name = f"blocks.{real_layer}.hook_resid_post"

    # захватываем переменные по значению — иначе замыкание сломается в цикле
    def steering_hook(act, hook, _layer=real_layer, _f=feature_index, _s=scale):
        return steer_sae_feature_all_positions(act, hook, _layer, _f, _s)

    with torch.no_grad():
        if scale == 0.0:
            out = generate_safely(prompt, max_new_tokens, do_sample, temperature)
        else:
            with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                out = generate_safely(prompt, max_new_tokens, do_sample, temperature)

    return out

# ====================== ОСНОВНОЙ ЦИКЛ ======================
steering_rows = []

total = len(TEST_TASKS) * len(STEERING_FEATURES) * len(STEERING_SCALES) * N_SAMPLES
print(f"Всего прогонов: {total}")

for task_id, task in enumerate(TEST_TASKS):
    full_prompt = build_analysis_prompt(BASE_TEXT, task)

    print(f"\n{'#'*60}")
    print(f"### TASK {task_id}: {task.strip()[:60]}")
    print(f"{'#'*60}")

    for real_layer, feature_index in STEERING_FEATURES:
        for scale in STEERING_SCALES:
            for sample_id in range(N_SAMPLES):

                tag = (f"TASK {task_id} | FEAT {real_layer}/{feature_index} "
                       f"| SCALE {scale:+.1f} | SAMPLE {sample_id}")
                print(f"\n=== {tag} ===")

                row = {
                    "task_id":       task_id,
                    "task":          task.strip(),
                    "real_layer":    real_layer,
                    "feature_index": feature_index,
                    "scale":         scale,
                    "sample_id":     sample_id,
                    "do_sample":     DO_SAMPLE,
                    "temperature":   TEMPERATURE,
                    "output":        "",
                    "error":         "",
                }

                try:
                    output = generate_with_feature_steering(
                        prompt=full_prompt,
                        real_layer=real_layer,
                        feature_index=feature_index,
                        scale=scale,
                    )
                    print(output[-600:])   # печатаем только хвост чтобы не спамить
                    row["output"] = output

                except Exception as e:
                    print(f"ERROR: {repr(e)}")
                    row["error"] = repr(e)

                steering_rows.append(row)

                # чистим VRAM после каждого прогона
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# ====================== СОХРАНЕНИЕ ======================
steering_df = pd.DataFrame(steering_rows)
steering_df.to_csv("sae_feature_steering_generation_test_sampled.csv", index=False)
print("\nСохранено в sae_feature_steering_generation_test_sampled.csv")
print(f"Итого строк: {len(steering_df)}")