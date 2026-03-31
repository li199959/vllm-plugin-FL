from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_mm_encoder_attention() -> None:
    """vLLM 0.16 already routes MM encoder attention through v1 backends.

    Keep this function as a no-op compatibility hook so older plugin call sites
    continue to work without pulling in removed private modules.
    """

    logger.debug_once(
        "MM encoder attention compatibility patch is a no-op on vLLM 0.16.",
        scope="local",
    )
