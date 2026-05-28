import os


def dsa_cp_env_enabled() -> bool:
    return os.getenv("VLLM_FL_ENABLE_DSA_CP", "0") == "1"


def is_deepseek_v32() -> bool:
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    return hasattr(vllm_config.model_config.hf_text_config, "index_topk")
