from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_mm_encoder_attention() -> None:
    """Compatibility hook for older FL call sites.

    vLLM 0.19 already routes MM encoder attention through the v1 attention
    stack, so the old monkey-patch target no longer exists.
    """

    logger.debug_once(
        "MM encoder attention compatibility patch is a no-op on vLLM 0.19.",
        scope="local",
    )
