"""Utils for evaluating the OpenVLA policy."""

import inspect
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure this repo's prismatic package wins over openvla-oft_code / site-packages copies.
_OPENVLA_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _OPENVLA_REPO_ROOT not in sys.path:
    sys.path.insert(0, _OPENVLA_REPO_ROOT)

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    model_config = OpenVLAConfig.from_pretrained(cfg.pretrained_checkpoint)
    # HF checkpoints auto_map to hub remote code that lacks output-stats support.
    if getattr(model_config, "auto_map", None):
        model_config.auto_map = None

    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        config=model_config,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
    )
    print(f"Loaded model: {type(vla)} ({OpenVLAForActionPrediction.__module__})")

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def aggregate_episode_output_stats(query_stats: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Aggregate per-query action-token output stats into one episode-level record."""
    if not query_stats:
        return None

    num_queries = len(query_stats)
    maxprob = float(np.exp(np.mean([np.log(max(item["maxprob"], 1e-12)) for item in query_stats])))
    nll = float(np.mean([item["nll"] for item in query_stats]))
    ppl = float(np.exp(nll))
    entropy = float(np.mean([item["entropy"] for item in query_stats]))
    ln_entropy = float(np.mean([item["ln_entropy"] for item in query_stats]))
    num_action_tokens = float(np.sum([item.get("num_action_tokens", 0.0) for item in query_stats]))

    return {
        "maxprob": maxprob,
        "nll": nll,
        "ppl": ppl,
        "entropy": entropy,
        "ln_entropy": ln_entropy,
        "num_action_tokens": num_action_tokens,
        "num_queries": float(num_queries),
        "score_ppl": float(1.0 / ppl),
        "score_entropy": float(-entropy),
        "score_ln_entropy": float(-ln_entropy),
    }


def append_output_stats_jsonl(record: Dict[str, Any], jsonl_path: str) -> None:
    """Append one episode-level output-stats record to JSONL."""
    if not jsonl_path:
        return
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_self_eval_output_dir(
    cfg,
    date_stamp: Optional[str] = None,
    run_stamp: Optional[str] = None,
) -> str:
    """Resolve output directory for self-eval JSON artifacts."""
    if getattr(cfg, "self_eval_output_dir", ""):
        return cfg.self_eval_output_dir

    date_stamp = date_stamp or os.getenv("DATE_STAMP") or DATE
    run_stamp = run_stamp or os.getenv("RUN_STAMP") or DATE_TIME
    checkpoint_name = str(cfg.pretrained_checkpoint).split("/")[-1]
    benchmark = getattr(cfg, "libero_benchmark", "libero")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_root = os.getenv(
        "OPENVLA_EVAL_RESULTS_ROOT",
        os.path.join(repo_root, "eval_results"),
    )
    return os.path.join(
        results_root,
        benchmark,
        checkpoint_name,
        cfg.task_suite_name,
        date_stamp,
        run_stamp,
    )


def write_self_eval_json(payload: Dict[str, Any], json_path: str) -> None:
    """Write self-eval scores to a JSON file."""
    if not json_path:
        return
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
    output_attentions=False,
    return_output_stats=False,
    action_sampling_temperature: float = 0.0,
    return_action_token_ids: bool = False,
):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    do_sample = action_sampling_temperature > 0
    predict_kwargs = {"do_sample": do_sample}
    if do_sample:
        predict_kwargs["temperature"] = float(action_sampling_temperature)

    predict_params = inspect.signature(vla.predict_action).parameters
    if "output_attentions" in predict_params:
        predict_kwargs["output_attentions"] = output_attentions
    if return_output_stats:
        if "return_output_stats" not in predict_params:
            raise RuntimeError(
                "Loaded OpenVLA model does not support return_output_stats. "
                "Ensure local openvla modeling code is used (not hub auto_map remote code)."
            )
        predict_kwargs["return_output_stats"] = True
    if return_action_token_ids:
        if "return_action_token_ids" not in predict_params:
            raise RuntimeError(
                "Loaded OpenVLA model does not support return_action_token_ids. "
                "Ensure local openvla modeling code is used (not hub auto_map remote code)."
            )
        predict_kwargs["return_action_token_ids"] = True

    result = vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        **predict_kwargs,
    )

    output_stats = None
    if return_output_stats and output_attentions:
        action, attentions, output_stats = result
    elif return_output_stats:
        action, output_stats = result
    elif output_attentions:
        action, attentions = result
    else:
        return result

    if output_attentions:
        num_prompt_tokens = int(inputs["input_ids"].shape[1])
        if not torch.all(inputs["input_ids"][:, -1] == 29871):
            num_prompt_tokens += 1

        num_patches = None
        if attentions is not None and len(attentions) > 0 and len(attentions[-1]) > 0:
            seq_len = int(attentions[-1][0].shape[-1]) + 1
            generated_so_far = len(attentions)
            inferred_num_patches = seq_len - num_prompt_tokens - generated_so_far
            if inferred_num_patches >= 0:
                num_patches = inferred_num_patches

        if return_output_stats:
            return action, attentions, num_patches, num_prompt_tokens, output_stats
        return action, attentions, num_patches, num_prompt_tokens

    return action, output_stats
