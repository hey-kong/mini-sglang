import time
import random

from transformers import AutoTokenizer
from minisgl.core import SamplingParams
from minisgl.llm import LLM

MODEL = "/data/llm/Llama-3.1-8B-Instruct"

PAGE_SIZE = 16

HOT_AWARE_HIRADIX = True

# align the hyperparameters
llm = LLM(
    MODEL,
    page_size=PAGE_SIZE,
    memory_ratio=0.9,
    hicache_ratio=4.0,
    device_mem_layout="page_first",
    host_mem_layout="page_first",
    cache_type="hiradix",
    use_layerwise=False,
    enable_hot_aware_hiradix=HOT_AWARE_HIRADIX,
)

sampling_params = SamplingParams(
    max_tokens=1,
    ignore_eos=True,
)

# -----------------------------
# Tokenizer: get safe token ids
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 可自行调大/调小
iterations_count = 20

def get_safe_token_ids(tok, need_count: int):
    vocab_size = tok.vocab_size
    special_ids = set()
    if tok.bos_token_id is not None:
        special_ids.add(tok.bos_token_id)
    if tok.eos_token_id is not None:
        special_ids.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        special_ids.add(tok.pad_token_id)
    if tok.unk_token_id is not None:
        special_ids.add(tok.unk_token_id)

    safe = []
    for i in range(10, vocab_size):
        if i not in special_ids:
            safe.append(i)
        if len(safe) >= need_count:
            break

    if len(safe) < need_count:
        raise RuntimeError(
            f"Cannot find enough safe token ids from tokenizer. "
            f"need={need_count}, got={len(safe)}"
        )
    return safe

NEED_SAFE_COUNT = 16 + iterations_count + 32
SAFE_TOKENS = get_safe_token_ids(tokenizer, NEED_SAFE_COUNT)
print("Using safe token ids:", SAFE_TOKENS[:10])

print(f"iterations_count: {iterations_count}")

# -----------------------------
# Helpers
# -----------------------------
def round_hit_len(prompt_size: int, hit_ratio: float, align: int):
    hit_len = int(prompt_size * hit_ratio)
    hit_len = (hit_len // align) * align
    hit_len = max(0, min(hit_len, prompt_size))
    return hit_len


def build_prefix_tokens(prefix_len: int, group_tag: int):
    """
    为每个测试组构造独立 prefix，避免不同组之间互相污染。
    """
    if prefix_len <= 0:
        return []

    x = SAFE_TOKENS[group_tag % len(SAFE_TOKENS)]
    y = SAFE_TOKENS[(group_tag + 1) % len(SAFE_TOKENS)]

    # 用交替模式，而不是全一样，降低跨组前缀碰撞概率
    out = []
    for i in range(prefix_len):
        out.append(x if (i % 2 == 0) else y)
    return out


def build_suffix_tokens_exact_hit(
    suffix_len: int,
    iter_id: int,
    group_tag: int,
    unique_tokens: list[int],
):
    """
    构造 suffix，并保证:
    - suffix[0] 在同组历史请求中唯一
    - 因而最长公共前缀严格停在 hit_len
    """
    if suffix_len <= 0:
        return []

    # 关键：第一个 suffix token 必须唯一
    first = unique_tokens[iter_id]

    # 后面怎么填都行，只要别全都一样
    a = SAFE_TOKENS[(group_tag + 3) % len(SAFE_TOKENS)]
    b = SAFE_TOKENS[(group_tag + 5) % len(SAFE_TOKENS)]
    c = SAFE_TOKENS[(group_tag + 7) % len(SAFE_TOKENS)]

    out = [first]
    for k in range(1, suffix_len):
        m = k % 3
        if m == 0:
            out.append(a)
        elif m == 1:
            out.append(b)
        else:
            out.append(c)
    return out


def build_full_miss_prompt(
    prompt_size: int,
    iter_id: int,
    unique_tokens: list[int],
):
    """
    构造 full miss baseline。
    关键：第 0 个 token 唯一，确保不会命中历史前缀。
    """
    if prompt_size <= 0:
        return []

    first = unique_tokens[iter_id]

    a = SAFE_TOKENS[1 % len(SAFE_TOKENS)]
    b = SAFE_TOKENS[2 % len(SAFE_TOKENS)]
    c = SAFE_TOKENS[3 % len(SAFE_TOKENS)]

    out = [first]
    for k in range(1, prompt_size):
        m = k % 3
        if m == 0:
            out.append(a)
        elif m == 1:
            out.append(b)
        else:
            out.append(c)
    return out


def run_once(token_ids):
    t0 = time.perf_counter()
    _ = llm.generate([token_ids], sampling_params)
    return time.perf_counter() - t0

assert iterations_count <= len(SAFE_TOKENS) - 16, \
    "SAFE_TOKENS 不够，无法严格保证每轮 suffix[0] 或 prompt[0] 唯一"

UNIQUE_TOKENS = SAFE_TOKENS[16:16 + iterations_count]

PROMPT_SIZES_TO_TEST = [4096]
HIT_RATIOS = [i / 10 for i in range(1, 11)]
BLOCK_ALIGN = 16

print("\n===== HBM hit + prefill =====")

results = []

for prompt_size in PROMPT_SIZES_TO_TEST:
    for ratio_idx, ratio in enumerate(HIT_RATIOS):
        hit_len = round_hit_len(prompt_size, ratio, BLOCK_ALIGN)
        miss_len = prompt_size - hit_len
        actual_ratio = hit_len / prompt_size if prompt_size > 0 else 0.0

        group_tag = prompt_size * 1000 + ratio_idx * 17

        prefix_tokens = build_prefix_tokens(hit_len, group_tag=group_tag)

        # 先把“恰好 hit_len 的 prefix”放进 cache
        if hit_len > 0:
            _ = run_once(prefix_tokens)

        total = 0.0
        for j in range(1):
            suffix_tokens = build_suffix_tokens_exact_hit(
                suffix_len=miss_len,
                iter_id=j,
                group_tag=group_tag,
                unique_tokens=UNIQUE_TOKENS,
            )
            prompt_tokens = prefix_tokens + suffix_tokens
            total += run_once(prompt_tokens)

        avg_ms = 1000 * total

        results.append({
            "prompt_size": prompt_size,
            "target_hit_ratio": ratio,
            "actual_hit_len": hit_len,
            "miss_len": miss_len,
            "actual_hit_ratio": actual_ratio,
            "avg_latency": avg_ms,
        })

# 统一最后打印
for prompt_size in PROMPT_SIZES_TO_TEST:
    print(f"\n--- prompt_size = {prompt_size} ---")
    for r in results:
        if r["prompt_size"] == prompt_size:
            print(
                f"target_hit_ratio={r['target_hit_ratio']:.1f}, "
                f"actual_hit_len={r['actual_hit_len']}, "
                f"miss_len={r['miss_len']}, "
                f"actual_hit_ratio={r['actual_hit_ratio']:.4f}, "
                f"avg_latency={r['avg_latency']:.2f} ms"
            )
