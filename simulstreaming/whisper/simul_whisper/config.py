# This code was originally in simul_whisper/transcriber/simul_whisper.py . It is adapted a lot for SimulStreaming.

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class SimulWhisperConfig:
    '''Options that are common for all simul policies that could be implemented in SimulWhisper.'''
    model_path: str
    language: str = field(default="zh")
    nonspeech_prob: float = 1.0
    audio_min_len: float = 1.0
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 5
    task: Literal["transcribe","translate"] = "transcribe"
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)

    logdir: str = field(default="logdir", metadata={"help": "Directory to save audio segments and tokens for debugging purposes."})
    
    # Performance optimization options
    use_compile: bool = field(default=True, metadata={"help": "Use torch.compile() for faster inference (PyTorch 2.0+). Can provide 10-30% speedup."})
    use_half_precision: bool = field(default=None, metadata={"help": "Use float16/bfloat16 for faster inference on GPU. None = auto-detect based on device."})
    enable_cudnn_benchmark: bool = field(default=True, metadata={"help": "Enable cuDNN benchmarking for consistent input sizes (can speed up on GPU)."})

@dataclass
class AlignAttConfig(SimulWhisperConfig):
    '''Options specific to the AlignAtt policy.'''
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 200 # in frames. Max value is 1500. Higher value turns rewinds off.
    audio_max_len: float = 30.0
    cif_ckpt_path: str = ""
    never_fire: bool = False