"""Weight storage presets with selectable BF16 or FP16 floating-point compute."""
from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class WeightFormat:
    id: str
    backend: str
    label: str
    bits: int
    quant_type: str = ""
    double_quant: bool = False


FORMATS = (
    WeightFormat("none", "none", "Floating point (16-bit)", 16),
    WeightFormat("bnb-int8", "bitsandbytes", "INT8 (8-bit, weight-only)", 8),
    WeightFormat("bnb-nf4", "bitsandbytes", "NF4 (4-bit)", 4, "nf4"),
    WeightFormat("bnb-fp4", "bitsandbytes", "FP4 (4-bit)", 4, "fp4"),
    WeightFormat("bnb-nf4-dq", "bitsandbytes", "NF4 (4-bit, double quant)", 4, "nf4", True),
    WeightFormat("bnb-fp4-dq", "bitsandbytes", "FP4 (4-bit, double quant)", 4, "fp4", True),
    *(WeightFormat(f"hqq-{bits}", "hqq", f"INT{bits} ({bits}-bit)", bits) for bits in (8, 4, 3, 2, 1)),
)


COMPUTE_DTYPES = ("bfloat16", "float16")


def resolve_compute_dtype(name):
    if name not in COMPUTE_DTYPES:
        raise ValueError(f"Unknown compute precision: {name!r}. Choose bfloat16 or float16.")
    import torch
    return getattr(torch, name)


def weight_format(preset):
    for item in FORMATS:
        if item.id == preset:
            return item
    raise ValueError(f"Unknown weight format: {preset!r}")


def validate_environment(preset, device):
    fmt = weight_format(preset)
    if fmt.backend == "none":
        return
    # Keep this application's supported quantized path explicit, including on Windows.
    if device != "cuda":
        raise ValueError("Quantized loading requires an NVIDIA CUDA GPU in this explorer. Use floating-point weights on CPU/MPS.")
    if find_spec(fmt.backend) is None:
        raise ValueError(f"Missing {fmt.backend}. Install with: uv sync --extra {fmt.backend}")


def make_quantization_config(preset, compute_dtype="bfloat16"):
    dtype = resolve_compute_dtype(compute_dtype)
    fmt = weight_format(preset)
    if fmt.backend == "none":
        return None
    from transformers import BitsAndBytesConfig, HqqConfig

    # The output head stays floating point for the lenses. Conv/norm layers are not Linear and are unchanged.
    skip = ["lm_head", "visual"]
    if fmt.backend == "bitsandbytes":
        return BitsAndBytesConfig(
            load_in_8bit=fmt.bits == 8,
            load_in_4bit=fmt.bits == 4,
            llm_int8_skip_modules=skip,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=fmt.quant_type or "nf4",
            bnb_4bit_use_double_quant=fmt.double_quant,
        )
    # Pure PyTorch backend works on Windows without compiling optional HQQ CUDA extensions.
    from hqq.core.quantize import HQQBackend, HQQLinear
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    return HqqConfig(nbits=fmt.bits, group_size=64, axis=1, skip_modules=skip)


def linear_weight(module):
    """Return an unpacked [out, in] weight only for the current operation; never keep a dense model copy."""
    import torch

    if callable(getattr(module, "dequantize", None)) and hasattr(module, "W_q"):
        return module.dequantize()  # HQQLinear has no ordinary .weight parameter
    weight = module.weight
    if getattr(weight, "quant_state", None) is not None:
        from bitsandbytes.functional import dequantize_4bit
        return dequantize_4bit(weight.data, weight.quant_state)
    if weight.dtype == torch.int8:
        from bitsandbytes.functional import int8_vectorwise_dequant
        scale = getattr(weight, "SCB", None)
        if scale is None:
            scale = getattr(getattr(module, "state", None), "SCB", None)
        if scale is None:
            raise RuntimeError("bitsandbytes INT8 weight is missing its row scales (SCB)")
        return int8_vectorwise_dequant(weight.data, scale).to(getattr(module, "compute_dtype", torch.bfloat16))
    return weight


def _int8_weight_only_forward(self, x):
    # Linear8bitLt's usual LLM.int8 forward also quantizes activations. Deliberately use floating-point
    # matmul with a temporary dequantized weight so this preset is strictly weight-only, too.
    import torch.nn.functional as F
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, linear_weight(self).to(x.dtype), bias)


def prepare_weight_only_model(model, preset, compute_dtype="bfloat16"):
    if preset == "bnb-int8":
        from types import MethodType
        from bitsandbytes.nn import Linear8bitLt
        for module in model.modules():
            if isinstance(module, Linear8bitLt):
                module.compute_dtype = resolve_compute_dtype(compute_dtype)
                module.forward = MethodType(_int8_weight_only_forward, module)
