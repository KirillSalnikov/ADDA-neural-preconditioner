#!/usr/bin/env python3
"""Export ConvSAI_Spectral checkpoints to ADDA .precond files."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from core.fft_matvec import FFTMatVec
from neural_precond.model import ConvSAI_Spectral, positions_to_occupancy
from apps.export_universal_precond import make_shape_positions, export_convsai_fft


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError("checkpoint does not contain a state_dict")
    if any(k.startswith("module.") for k in obj):
        obj = {k.removeprefix("module."): v for k, v in obj.items()}
    obj = _translate_legacy_spectral_keys(obj)
    return obj


def _translate_legacy_spectral_keys(state):
    if any(k.startswith("freq_mlp.blocks.") for k in state):
        return state
    if "freq_mlp.proj_in.weight" not in state:
        return state

    translated = {}
    extra_ids = sorted({
        int(k.split(".")[2])
        for k in state
        if k.startswith("freq_mlp.extra.") and k.endswith(".weight")
    })
    out_idx = 2 * (len(extra_ids) + 1)

    for key, value in state.items():
        if key.startswith("freq_mlp.proj_in."):
            key = key.replace("freq_mlp.proj_in.", "freq_mlp.0.")
        elif key.startswith("freq_mlp.extra."):
            parts = key.split(".")
            extra_idx = int(parts[2])
            seq_idx = 2 + 2 * extra_idx
            key = f"freq_mlp.{seq_idx}.{parts[3]}"
        elif key.startswith("freq_mlp.proj_out."):
            key = key.replace("freq_mlp.proj_out.", f"freq_mlp.{out_idx}.")
        translated[key] = value
    return translated


def _infer_config_path(checkpoint):
    folder_config = os.path.join(os.path.dirname(checkpoint), "config.json")
    parent_config = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), "config.json")
    for path in (folder_config, parent_config):
        if os.path.exists(path):
            return path
    if checkpoint.endswith(os.path.join("models", "spectral", "checkpoints", "best_model.pt")):
        legacy = os.path.join("results", "spectral_v3_adda", "config.json")
        if os.path.exists(legacy):
            return legacy
    return None


def _load_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_arch_from_state(state):
    config = {}
    if "freq_mlp.proj_in.weight" in state:
        config["freq_hidden"] = int(state["freq_mlp.proj_in.weight"].shape[0])
        freq_input = int(state["freq_mlp.proj_in.weight"].shape[1])
        config["no_freq_coords"] = (freq_input == 18 + config["freq_hidden"])
        block_ids = sorted({
            int(k.split(".")[2])
            for k in state
            if k.startswith("freq_mlp.blocks.") and k.endswith(".0.weight")
        })
        if block_ids:
            config["freq_residual_blocks"] = len(block_ids)
    elif "freq_mlp.0.weight" in state:
        config["freq_hidden"] = int(state["freq_mlp.0.weight"].shape[0])
        freq_input = int(state["freq_mlp.0.weight"].shape[1])
        config["no_freq_coords"] = (freq_input == 18 + config["freq_hidden"])
    if "global_enc.0.weight" in state:
        config["global_hidden"] = int(state["global_enc.0.weight"].shape[0])
    if "shape_encoder.fc.weight" in state:
        config["shape_embed_dim"] = int(state["shape_encoder.fc.weight"].shape[0])
    if any(k.startswith("correction_mlp.") for k in state):
        corr_weights = [
            (int(k.split(".")[1]), v)
            for k, v in state.items()
            if k.startswith("correction_mlp.") and k.endswith(".weight")
        ]
        corr_weights.sort(key=lambda x: x[0])
        if corr_weights:
            config["spectral_correction_hidden"] = int(corr_weights[0][1].shape[0])
            config["spectral_correction_layers"] = len(corr_weights)
        if "_correction_scale" in state:
            config["spectral_correction_scale"] = float(state["_correction_scale"])
    if any(k.startswith("coarse_transformer.") for k in state):
        config["spectral_transformer_tokens"] = 8
        if "coarse_transformer.token_proj.weight" in state:
            config["spectral_transformer_dim"] = int(
                state["coarse_transformer.token_proj.weight"].shape[0])
        layer_ids = sorted({
            int(k.split(".")[3])
            for k in state
            if k.startswith("coarse_transformer.encoder.layers.")
            and k.endswith("self_attn.in_proj_weight")
        })
        if layer_ids:
            config["spectral_transformer_layers"] = len(layer_ids)

    freq_weights = [k for k in state if k.startswith("freq_mlp.") and k.endswith(".weight")]
    global_weights = [k for k in state if k.startswith("global_enc.") and k.endswith(".weight")]
    if freq_weights:
        config["freq_layers"] = len(freq_weights)
    if global_weights:
        config["global_layers"] = len(global_weights)

    channels = []
    for idx in (0, 3, 6):
        key = f"shape_encoder.conv.{idx}.weight"
        if key in state:
            channels.append(int(state[key].shape[0]))
    if channels:
        config["encoder_channels"] = channels
    return config


def _build_model(config, state, args):
    inferred = _infer_arch_from_state(state)
    merged = {**inferred, **config}

    def pick(name, default):
        value = getattr(args, name, None)
        return value if value is not None else merged.get(name, default)

    normalize_inputs = merged.get("spectral_normalize_inputs", False)
    if args.normalize_inputs is not None:
        normalize_inputs = args.normalize_inputs

    model = ConvSAI_Spectral(
        freq_hidden=int(pick("freq_hidden", 64)),
        freq_layers=int(pick("freq_layers", 3)),
        global_hidden=int(pick("global_hidden", 256)),
        global_layers=int(pick("global_layers", 3)),
        shape_embed_dim=int(pick("shape_embed_dim", 16)),
        activation=merged.get("activation", "relu"),
        squared=bool(merged.get("squared_kernel", True)),
        freq_coords=not bool(merged.get("no_freq_coords", False)),
        freq_residual_blocks=int(merged.get("freq_residual_blocks", 0)),
        correction_hidden=int(pick("spectral_correction_hidden", 0)),
        correction_layers=int(pick("spectral_correction_layers", 2)),
        correction_scale=float(pick("spectral_correction_scale", 1.0)),
        transformer_tokens=int(pick("spectral_transformer_tokens", 0)),
        transformer_dim=int(pick("spectral_transformer_dim", 128)),
        transformer_layers=int(pick("spectral_transformer_layers", 2)),
        transformer_heads=int(pick("spectral_transformer_heads", 4)),
        transformer_scale=float(pick("spectral_transformer_scale", 1.0)),
        encoder_resolution=int(merged.get("encoder_resolution", 32)),
        encoder_channels=tuple(merged.get("encoder_channels", [16, 32, 64])),
        normalize_inputs=bool(normalize_inputs),
        m_re_min=float(merged.get("m_re_min", 1.5)),
        m_re_max=float(merged.get("m_re_max", 4.0)),
        m_im_min=float(merged.get("m_im_min", 0.0)),
        m_im_max=float(merged.get("m_im_max", 0.5)),
        kd_min=float(merged.get("kd_min", 0.2)),
        kd_max=float(merged.get("kd_max", 0.8)),
        log_grid_center=float(merged.get("spectral_log_grid_center", 2.0)),
        log_grid_scale=float(merged.get("spectral_log_grid_scale", 2.0)),
    )
    model.load_state_dict(state)
    if getattr(args, "spectral_correction_scale", None) is not None:
        scale_buffer = getattr(model, "_correction_scale", None)
        if scale_buffer is None:
            raise ValueError("--spectral_correction_scale requires a checkpoint with correction_mlp")
        scale_buffer.fill_(float(args.spectral_correction_scale))
        merged["spectral_correction_scale"] = float(args.spectral_correction_scale)
    model.eval()
    return model, merged


def _shape_args(shape, ay, az):
    if shape in ("hex", "hex_prism"):
        return "prism", 6.0, az
    return shape, ay, az


def _symmetry_transforms(name):
    name = (name or "none").lower()
    ident = np.eye(3, dtype=np.int32)
    if name in ("none", "off", "false"):
        return [ident]
    transforms = [ident]
    if name in ("z180", "z180_zflip", "d2"):
        transforms.append(np.diag([-1, -1, 1]).astype(np.int32))
    if name in ("zflip", "z180_zflip", "d2"):
        transforms.append(np.diag([1, 1, -1]).astype(np.int32))
    if name in ("z180_zflip", "d2"):
        transforms.append(np.diag([-1, -1, -1]).astype(np.int32))
    if len(transforms) == 1:
        raise ValueError(f"unknown --symmetry={name!r}; use none, z180, zflip, or z180_zflip")
    return transforms


def _symmetrized_block(M_spatial, di, dj, dk, transforms):
    if len(transforms) == 1:
        return M_spatial[:, :, di % M_spatial.shape[2],
                         dj % M_spatial.shape[3],
                         dk % M_spatial.shape[4]]
    gx, gy, gz = M_spatial.shape[2:]
    offset = np.array([di, dj, dk], dtype=np.int32)
    acc = np.zeros((3, 3), dtype=np.complex128)
    for transform in transforms:
        src = transform.T @ offset
        block = M_spatial[:, :, src[0] % gx, src[1] % gy, src[2] % gz]
        acc += transform @ block @ transform.T
    return acc / len(transforms)


def _extract_stencil(M_hat, threshold_rel, max_radius, symmetry="none"):
    M_hat_np = M_hat.detach().cpu().numpy().astype(np.complex128)
    M_spatial = np.fft.ifftn(M_hat_np, axes=(2, 3, 4))
    transforms = _symmetry_transforms(symmetry)

    gx, gy, gz = M_spatial.shape[2:]
    max_abs = float(np.max(np.abs(M_spatial)))
    threshold = threshold_rel * max_abs

    if max_radius is None:
        ranges = (range(-(gx // 2), gx // 2),
                  range(-(gy // 2), gy // 2),
                  range(-(gz // 2), gz // 2))
    else:
        r = int(max_radius)
        ranges = (range(-r, r + 1), range(-r, r + 1), range(-r, r + 1))

    stencil, kernel = [], []
    for di in ranges[0]:
        for dj in ranges[1]:
            for dk in ranges[2]:
                block = _symmetrized_block(M_spatial, di, dj, dk, transforms)
                if np.max(np.abs(block)) > threshold:
                    stencil.append([di, dj, dk])
                    kernel.append(block)

    if not stencil:
        raise RuntimeError("export produced an empty stencil; lower --threshold-rel")
    return np.array(stencil, dtype=np.int32), np.array(kernel, dtype=np.complex128), max_abs


def _blend_with_identity(M_hat, blend):
    blend = float(blend)
    if blend == 1.0:
        return M_hat
    if blend < 0.0 or blend > 1.0:
        raise ValueError("--blend-identity must be in [0, 1]")
    out = M_hat * blend
    eye = torch.eye(3, dtype=out.dtype, device=out.device)
    out = out + (1.0 - blend) * eye[:, :, None, None, None]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--shape", default="sphere")
    parser.add_argument("--ay", type=float, default=1.0)
    parser.add_argument("--az", type=float, default=1.0)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--m_re", type=float, default=3.0)
    parser.add_argument("--m_im", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.41887902047863906)
    parser.add_argument("--threshold-rel", type=float, default=1e-8)
    parser.add_argument("--max-radius", type=int, default=None)
    parser.add_argument("--blend-identity", type=float, default=1.0,
                        help="Export (1-lambda)*I + lambda*M_hat before spatial thresholding")
    parser.add_argument("--symmetry", default="none",
                        choices=["none", "z180", "zflip", "z180_zflip", "d2"],
                        help="Average exported spatial kernel over cheap prism symmetries")
    parser.add_argument("--normalize-inputs", action="store_true", default=None)
    parser.add_argument("--freq_hidden", type=int, default=None)
    parser.add_argument("--freq_layers", type=int, default=None)
    parser.add_argument("--spectral-correction-hidden", "--spectral_correction_hidden",
                        dest="spectral_correction_hidden", type=int, default=None)
    parser.add_argument("--spectral-correction-layers", "--spectral_correction_layers",
                        dest="spectral_correction_layers", type=int, default=None)
    parser.add_argument("--spectral-correction-scale", "--spectral_correction_scale",
                        dest="spectral_correction_scale", type=float, default=None)
    parser.add_argument("--spectral-transformer-tokens", "--spectral_transformer_tokens",
                        dest="spectral_transformer_tokens", type=int, default=None)
    parser.add_argument("--spectral-transformer-dim", "--spectral_transformer_dim",
                        dest="spectral_transformer_dim", type=int, default=None)
    parser.add_argument("--spectral-transformer-layers", "--spectral_transformer_layers",
                        dest="spectral_transformer_layers", type=int, default=None)
    parser.add_argument("--spectral-transformer-heads", "--spectral_transformer_heads",
                        dest="spectral_transformer_heads", type=int, default=None)
    parser.add_argument("--spectral-transformer-scale", "--spectral_transformer_scale",
                        dest="spectral_transformer_scale", type=float, default=None)
    parser.add_argument("--global_hidden", type=int, default=None)
    parser.add_argument("--global_layers", type=int, default=None)
    parser.add_argument("--shape_embed_dim", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    config_path = args.config or _infer_config_path(checkpoint)
    config = _load_config(config_path)

    state = _state_dict(_torch_load(checkpoint))
    model, merged = _build_model(config, state, args)

    shape, ay, az = _shape_args(args.shape, args.ay, args.az)
    positions = make_shape_positions(shape, args.grid, ay, az)
    occ = positions_to_occupancy(positions, grid_size=None, device="cpu")
    fft_mv = FFTMatVec(
        torch.tensor(positions, dtype=torch.long),
        1.0,
        complex(args.m_re, args.m_im),
        args.kd,
        "cpu",
    )

    with torch.no_grad():
        cond = model(args.m_re, args.m_im, args.kd, occ, args.grid)
        M_hat = model.build_M_hat(cond, fft_mv)
        M_hat = _blend_with_identity(M_hat, args.blend_identity)

    stencil, kernel, max_abs = _extract_stencil(
        M_hat, args.threshold_rel, args.max_radius, symmetry=args.symmetry)
    export_convsai_fft(stencil, kernel, len(positions), args.output)
    print(
        f"EXPORT SUCCESS: {args.output}. shape={shape} grid={args.grid} "
        f"N={len(positions)} stencil={len(stencil)} max_abs={max_abs:.4e} "
        f"blend={args.blend_identity:.4g} symmetry={args.symmetry} "
        f"config={config_path or 'inferred'} normalize={model.normalize_inputs} "
        f"freq_hidden={merged.get('freq_hidden')} freq_layers={merged.get('freq_layers')} "
        f"correction_scale={merged.get('spectral_correction_scale')} "
        f"transformer_tokens={merged.get('spectral_transformer_tokens', 0)}"
    )


if __name__ == "__main__":
    main()
