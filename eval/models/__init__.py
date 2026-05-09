__all__ = [
    "Qwen2_5VL",
    "Qwen3VL",
    "LlavaSTQwen2",
    "Llava16Model",
    "VideoMolmoModel",
    "VideoChatR1",
    "STVGR1",
]


def __getattr__(name):
    if name in {"Qwen2_5VL", "Qwen3VL"}:
        from .qwen_family import Qwen2_5VL, Qwen3VL

        return {"Qwen2_5VL": Qwen2_5VL, "Qwen3VL": Qwen3VL, "Qwen3.5": Qwen3VL}[name]

    if name == "LlavaSTQwen2":
        from .llava_st import LlavaSTQwen2

        return LlavaSTQwen2

    if name == "Llava16Model":
        from .llava_16 import Llava16Model

        return Llava16Model

    if name == "VideoMolmoModel":
        from .videomolmo import VideoMolmoModel

        return VideoMolmoModel

    if name == "VideoChatR1":
        from .videochat_r1 import VideoChatR1

        return VideoChatR1

    if name == "STVGR1":
        from .stvg_r1 import STVGR1

        return STVGR1

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
