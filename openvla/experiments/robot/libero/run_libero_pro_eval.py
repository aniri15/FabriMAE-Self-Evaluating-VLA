"""
run_libero_pro_eval.py — LIBERO-PRO evaluation for OpenVLA.

Note (MAE naming): user-facing metric MAE-D (self_eval_mode=mae-d) uses internal mad
computation (top-k visual entropy). MAE-C is PI-only.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import torch
import yaml

import draccus
from experiments.robot.libero.repo_paths import (
    default_evaluation_config_swap,
    libero_pro_libero_root,
    setup_libero_config,
)
import debugpy
import numpy as np
import tqdm
from libero.libero import benchmark
from typing import Any

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.mae_metric_naming import DEFAULT_MAE_D_OUTPUT, MAE_D_MODE
from experiments.robot.libero.attention_mad_utils import (
    append_online_mad_jsonl,
    build_episode_mad_record,
    compute_visual_entropy_lh_from_attentions,
    rollout_requests_attentions,
)
from experiments.robot.libero.self_eval_utils import (
    aggregate_episode_token_consistency,
    compute_step_token_consistency,
    configure_self_eval,
    self_eval_json_filename,
)
from experiments.robot.libero.LIBERO_PRO import perturbation
from experiments.robot.openvla_utils import (
    aggregate_episode_output_stats,
    get_processor,
    resolve_self_eval_output_dir,
    write_self_eval_json,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_action_with_token_ids,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    libero_benchmark: str = "libero_pro"             # libero | libero_pro
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    max_parallel_tasks: int = 2                      # Number of LIBERO tasks to evaluate concurrently (deprecated: now always sequential)
    task_start: int = -1                             # Start task ID (inclusive, -1 means start from 0)
    task_end: int = -1
    evaluation_config_path: str = ""  # set via default_evaluation_config_swap() in main
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    save_attentions: bool = False
    save_attn_dir: str = ''
    track_output_stats: bool = False                 # Save action-token output stats (maxprob/ppl/entropy)

    enable_self_eval: bool = True                    # Enable self-evaluation scoring during rollout
    self_eval_mode: str = "output_stats"             # consistency | output_stats | mae-d
    consistency_repeats_per_init: int = 3            # token_sample: samples per action step; outcome_repeat: full rollouts
    consistency_method: str = "token_sample"         # token_sample | outcome_repeat
    action_sampling_temperature: float = 0.0         # >0 enables token sampling (required for token_sample)
    self_eval_output_dir: str = ""                   # Optional override for self-eval JSON output dir
    attention_eval_method: str = "mae-d"             # mae-d only (MAE-C is PI-only)
    attention_eval_ratios: str = "0.01"              # Comma-separated top-k ratios for online MAE-D
    attention_eval_output_name: str = DEFAULT_MAE_D_OUTPUT

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on

def save_single_attention_weight(
    cfg: Any,
    saved_attentions: Any,
    task_id: int,
    episode_idx: int,
    query_idx: int,
    task_description: str,
    img: Any,
    success: bool,
    episode_number: int,
    log_file=None,
    num_patches: Optional[int] = None,
    num_prompt_tokens: Optional[int] = None,
    ):
    """
    极速版: 保存为无 Key 的 Compact Tensor 格式 (.pt)。
    结构: (Metadata_Tuple, Summary_Tensor, Chunk_Tensor, Token_Tensor)
    """
    if not cfg.save_attn_dir or saved_attentions is None:
        return

    try:
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        
        # --- Constants ---
        ACTION_DIM = 6

        # Validate inputs
        if num_patches is None or num_prompt_tokens is None:
            return

        # Indices
        visual_start, visual_end = 1, num_patches + 1
        text_start, text_end = num_patches + 1, num_patches + num_prompt_tokens
        action_start = text_end

        # Extract Attention Matrices
        attentions = [np.array([layer_attention[:, :, :, : text_end].float().cpu().numpy() for layer_attention in saved_attention]) for saved_attention in saved_attentions[1: ]]
        attentions = np.concatenate(attentions, axis=3).swapaxes(0, 1)[0] # (num_batch, num_heads, act_len, input_len)
        # Containers for stacking
        layers_summary = [] # List of (Head, 6, 8, 7)
        # layers_chunks = []  # List of (Head, 8, 8, 7)
        # layers_tokens = []  # List of (Head, 56, 8, 7)
        layers_visual = []
        valid_layer_indices = []

        # Helper: Entropy
        def batch_entropy(probs, axis=-1, epsilon=1e-12):
            row_sums = np.sum(probs, axis=axis, keepdims=True)
            probs_norm = probs / (row_sums + epsilon)
            probs_norm = np.clip(probs_norm, epsilon, 1.0)
            return -np.sum(probs_norm * np.log(probs_norm), axis=axis)

        # --- Processing Loop ---
        if isinstance(attentions, np.ndarray):
            for l, layer_attn in enumerate(attentions):
                if layer_attn is None:
                    continue
                
                # Move to CPU & Numpy
                if hasattr(layer_attn, 'cpu'):
                    layer_attn = layer_attn.float().cpu().numpy()
                else:
                    layer_attn = layer_attn.astype(np.float32)

                # layer_attn: (1, num_heads, seq_len, seq_len)
                num_heads, seq_len, _ = layer_attn.shape
                
                # Pre-allocate head arrays for this layer to avoid list append overhead
                # Summary: 6 metrics (VisSum, TxtSum, ActSum, VisEnt, TxtEnt, ActEnt)
                head_summary_stack = np.zeros((num_heads, 4, ACTION_DIM), dtype=np.float32)
                head_visual_stack = np.zeros((num_heads, ACTION_DIM, num_patches), dtype=np.float32)
                # Chunks: 8 chunks
                # head_chunk_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Tokens: 56 tokens
                #head_token_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)

                for h in range(num_heads):
                    # Extract the block of interest: (56, seq_len)
                    # Rows: The 56 action tokens generated
                    head_block = layer_attn[h, : , :]
                    head_visual_stack[h] = head_block
                    # 1. Summary Metrics (Vectorized)
                    # Sums: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_sum = head_block[:, visual_start:visual_end].sum(axis=1)
                    txt_sum = head_block[:, text_start:text_end].sum(axis=1)

                    
                    # Entropies: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_ent = batch_entropy(head_block[:, visual_start:visual_end])
                    txt_ent = batch_entropy(head_block[:, text_start:text_end])
                    

                    # Stack into (6, 8, 7)
                    head_summary_stack[h, 0] = vis_sum
                    head_summary_stack[h, 1] = txt_sum
                    head_summary_stack[h, 2] = vis_ent
                    head_summary_stack[h, 3] = txt_ent


                    # 2. Chunk Sums (Vectorized)
                    # Optimization: Extract the full action-to-action matrix (56, 56)
                    # (56, seq_len) -> (56, 56)
                    
                    # Reshape cols to (56, 8, 7) and sum over the last dim (dim 2)
                    # Result: (56, 8) where 8 is the chunk index
                    # But we need to store it as (8, 8, 7) where first 8 is chunk index
    


                # Store layer result
                layers_summary.append(head_summary_stack)
                layers_visual.append(head_visual_stack)
                # layers_chunks.append(head_chunk_stack)
                # layers_tokens.append(head_token_stack)
                valid_layer_indices.append(l)

                del layer_attn, head_block, head_summary_stack, head_visual_stack

        # --- Final Stacking & Saving ---
        if layers_summary:
            # 1. Stack Layers -> Tensor
            # Shape: (Num_Layers, Num_Heads, ...)
            # Using torch.from_numpy is zero-copy (mostly) and fast
            tensor_summary = torch.from_numpy(np.stack(layers_summary)) # (L, H, 6, 8, 7)
            tensor_visual = torch.from_numpy(np.stack(layers_visual))
            # tensor_chunks = torch.from_numpy(np.stack(layers_chunks))   # (L, H, 8, 8, 7)
            # tensor_tokens = torch.from_numpy(np.stack(layers_tokens))   # (L, H, 56, 8, 7)

            # 2. Prepare Metadata Tuple (Fixed Order)
            metadata = (
                task_id,            # 0
                episode_idx,        # 1
                query_idx,          # 2
                episode_number,     # 3
                success,            # 4
                task_description,   # 5
                valid_layer_indices,# 6 (List of ints)
                num_patches,        # 7
                num_prompt_tokens   # 8
            )

            # 3. Save as Tuple
            # file extension .pt
            save_path = os.path.join(cfg.save_attn_dir, f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}.pt")
            
            torch.save((metadata, tensor_summary, tensor_visual, img), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()

def save_single_attention_query(
    cfg: Any,
    saved_attentions: Any,
    task_id: int,
    episode_idx: int,
    query_idx: int,
    task_description: str,
    success: bool,
    episode_number: int,
    log_file=None,
    num_patches: Optional[int] = None,
    num_prompt_tokens: Optional[int] = None,
    ):
    """
    极速版: 保存为无 Key 的 Compact Tensor 格式 (.pt)。
    结构: (Metadata_Tuple, Summary_Tensor, Chunk_Tensor, Token_Tensor)
    """
    if not cfg.save_attn_dir or saved_attentions is None:
        return

    try:
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        
        # --- Constants ---
        ACTION_DIM = 6

        # Validate inputs
        if num_patches is None or num_prompt_tokens is None:
            return

        # Indices
        visual_start, visual_end = 1, num_patches + 1
        text_start, text_end = num_patches + 1, num_patches + num_prompt_tokens
        action_start = text_end

        # Extract Attention Matrices
        attentions = [np.array([layer_attention[:, :, :, : text_end].float().cpu().numpy() for layer_attention in saved_attention]) for saved_attention in saved_attentions[1: ]]
        attentions = np.concatenate(attentions, axis=3).swapaxes(0, 1)[0] # (num_batch, num_heads, act_len, input_len)
        # Containers for stacking
        layers_summary = [] # List of (Head, 6, 8, 7)
        # layers_chunks = []  # List of (Head, 8, 8, 7)
        # layers_tokens = []  # List of (Head, 56, 8, 7)
        valid_layer_indices = []

        # Helper: Entropy
        def batch_entropy(probs, axis=-1, epsilon=1e-12):
            row_sums = np.sum(probs, axis=axis, keepdims=True)
            probs_norm = probs / (row_sums + epsilon)
            probs_norm = np.clip(probs_norm, epsilon, 1.0)
            return -np.sum(probs_norm * np.log(probs_norm), axis=axis)

        # --- Processing Loop ---
        if isinstance(attentions, np.ndarray):
            for l, layer_attn in enumerate(attentions):
                if layer_attn is None:
                    continue
                
                # Move to CPU & Numpy
                if hasattr(layer_attn, 'cpu'):
                    layer_attn = layer_attn.float().cpu().numpy()
                else:
                    layer_attn = layer_attn.astype(np.float32)

                # layer_attn: (1, num_heads, seq_len, seq_len)
                num_heads, seq_len, _ = layer_attn.shape
                
                # Pre-allocate head arrays for this layer to avoid list append overhead
                # Summary: 6 metrics (VisSum, TxtSum, ActSum, VisEnt, TxtEnt, ActEnt)
                head_summary_stack = np.zeros((num_heads, 4, ACTION_DIM), dtype=np.float32)
                # Chunks: 8 chunks
                # head_chunk_stack = np.zeros((num_heads, NUM_ACTIONS_CHUNK, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
                # Tokens: 56 tokens
                #head_token_stack = np.zeros((num_heads, NUM_ACTION_TOKENS, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)

                for h in range(num_heads):
                    # Extract the block of interest: (56, seq_len)
                    # Rows: The 56 action tokens generated
                    head_block = layer_attn[h, : , :]

                    # 1. Summary Metrics (Vectorized)
                    # Sums: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_sum = head_block[:, visual_start:visual_end].sum(axis=1)
                    txt_sum = head_block[:, text_start:text_end].sum(axis=1)

                    
                    # Entropies: (56, modality_seq_len) -> Reshape (8, 7)
                    vis_ent = batch_entropy(head_block[:, visual_start:visual_end])
                    txt_ent = batch_entropy(head_block[:, text_start:text_end])
                    

                    # Stack into (6, 8, 7)
                    head_summary_stack[h, 0] = vis_sum
                    head_summary_stack[h, 1] = txt_sum
                    head_summary_stack[h, 2] = vis_ent
                    head_summary_stack[h, 3] = txt_ent


                    # 2. Chunk Sums (Vectorized)
                    # Optimization: Extract the full action-to-action matrix (56, 56)
                    # (56, seq_len) -> (56, 56)
                    
                    # Reshape cols to (56, 8, 7) and sum over the last dim (dim 2)
                    # Result: (56, 8) where 8 is the chunk index
                    # But we need to store it as (8, 8, 7) where first 8 is chunk index
    


                # Store layer result
                layers_summary.append(head_summary_stack)
                # layers_chunks.append(head_chunk_stack)
                # layers_tokens.append(head_token_stack)
                valid_layer_indices.append(l)

                del layer_attn, head_block, head_summary_stack 

        # --- Final Stacking & Saving ---
        if layers_summary:
            # 1. Stack Layers -> Tensor
            # Shape: (Num_Layers, Num_Heads, ...)
            # Using torch.from_numpy is zero-copy (mostly) and fast
            tensor_summary = torch.from_numpy(np.stack(layers_summary)) # (L, H, 6, 8, 7)
            # tensor_chunks = torch.from_numpy(np.stack(layers_chunks))   # (L, H, 8, 8, 7)
            # tensor_tokens = torch.from_numpy(np.stack(layers_tokens))   # (L, H, 56, 8, 7)

            # 2. Prepare Metadata Tuple (Fixed Order)
            metadata = (
                task_id,            # 0
                episode_idx,        # 1
                query_idx,          # 2
                episode_number,     # 3
                success,            # 4
                task_description,   # 5
                valid_layer_indices,# 6 (List of ints)
                num_patches,        # 7
                num_prompt_tokens   # 8
            )

            # 3. Save as Tuple
            # file extension .pt
            save_path = os.path.join(cfg.save_attn_dir, f"task_{task_id}_episode_{episode_idx}_query_{query_idx}_ep_{episode_number}.pt")
            
            torch.save((metadata, tensor_summary), save_path)

    except Exception as e:
        print(f"Error saving fast attention: {e}")
        import traceback
        traceback.print_exc()

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    configure_self_eval(cfg)
    if cfg.action_sampling_temperature > 0:
        torch.backends.cudnn.deterministic = False

    # Save original task_suite_name before it gets modified by evaluation config
    original_task_suite_name = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

        # Modify this section: Handle evaluation config for LIBERO-PRO perturbations
    if cfg.evaluation_config_path is not None:
        
        # [CRITICAL FIX] We need to redirect LIBERO to use our local project assets.
        # LIBERO looks for a config.yaml in the folder specified by LIBERO_CONFIG_PATH.
        # We will create a temporary config folder and file pointing to our local paths.
        
        config_dir = setup_libero_config(libero_pro_libero_root())
        print(f"[INFO] LIBERO_CONFIG_PATH set to: {config_dir}")

        with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
            evaluation_cfg = yaml.safe_load(f)

        evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
        evaluation_cfg["task_suite_name"] = cfg.task_suite_name

        use_swap = evaluation_cfg.get("use_swap", False)
        use_object = evaluation_cfg.get("use_object", False)
        use_language = evaluation_cfg.get("use_language", False)
        use_task = evaluation_cfg.get("use_task", False)
        use_environment = evaluation_cfg.get("use_environment", False)

        # Step 1: Check if only one of the use_xxx flags is True
        if sum([use_swap, use_object, use_language, use_task, use_environment]) > 1:
            # If more than one flag is True, use the temp environment
            bddl_file_path = evaluation_cfg.get("bddl_files_path", "") + cfg.task_suite_name + "_temp/"

            init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_temp/"

            # Check if the directories exist and the log.txt file contents match
            if not os.path.exists(bddl_file_path) or not os.path.exists(init_file_path):
                # If directories don't exist, create them and the log.txt file
                os.makedirs(init_file_path, exist_ok=True)
                os.makedirs(bddl_file_path, exist_ok=True)

                # Create the log.txt dynamically based on current flag values
                log_content = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
                with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                    log_file.write(log_content)  # Write the dynamic state to the log file

                perturbation.create_env(configs=evaluation_cfg)
            else:
                # If directories exist, check the contents of the log.txt file
                with open(os.path.join(bddl_file_path, "log.txt"), "r") as log_file:
                    log_contents = log_file.read().strip()

                # Define the expected log content based on the current flags
                expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"

                # If the log contents don't match, clean up and recreate the environment
                if log_contents != expected_log:
                    # Remove existing files in both directories
                    for folder in [bddl_file_path, init_file_path]:
                        for root, dirs, files in os.walk(folder, topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                    # Create the environment again
                    os.makedirs(init_file_path, exist_ok=True)
                    os.makedirs(bddl_file_path, exist_ok=True)

                    # Write the updated log content based on current flags
                    with open(os.path.join(bddl_file_path, "log.txt"), "w") as log_file:
                        log_file.write(expected_log)  # Write the updated log

                    perturbation.create_env(configs=evaluation_cfg)

            # Update task_suite_name with "_temp" suffix
            cfg.task_suite_name = cfg.task_suite_name + "_temp"

        # Step 2: Handle the case when only one use_xxx flag is True
        else:
            # [FIX] Generate a dynamic suffix based on the perturbation type
            # This ensures that different perturbation settings use different folders
            # reducing the need to manually delete the _temp folder.
            suffix = "_temp"
            if use_swap:
                suffix += "_swap"
            elif use_object:
                suffix += "_object"
            elif use_language:
                suffix += "_lang"
            elif use_task:
                suffix += "_task"
            elif use_environment:
                suffix += "_env"
            
            init_file_path = evaluation_cfg.get("init_file_dir", "") + cfg.task_suite_name + suffix

            if not os.path.exists(init_file_path):
                # Pass the suffix to create_env so it generates the correct folder name
                perturbation.create_env(configs=evaluation_cfg, suffix=suffix)

            cfg.task_suite_name = cfg.task_suite_name + suffix
    
    else:
        print("No evaluation config path provided, using original task suite name")

    # Initialize LIBERO task suite
    # [CRITICAL FIX] Register missing objects from assets
    # Many objects like 'yellow_plate', 'red_bowl' exist as XML files but are not registered in LIBERO's python code.
    # We need to dynamically register them so they can be used in perturbations.
    
    from libero.libero.envs.base_object import register_object, register_visual_change_object
    from libero.libero.envs.objects.google_scanned_objects import GoogleScannedObject
    from libero.libero.envs.objects.articulated_objects import ArticulatedObject
    # Base object class for dynamic registrations and manual mug classes
    from robosuite.models.objects import MujocoXMLObject
    # Force import turbosquid_objects so built-in mug objects are registered
    from libero.libero.envs.objects import turbosquid_objects  # noqa: F401
    from libero.libero.envs import objects
    # import os  <-- REMOVED: os is already imported globally
    
    # Define the path to assets
    # We use the project_libero_root we defined earlier
    if 'project_libero_root' in locals():
        assets_root = os.path.join(project_libero_root, "assets")
    else:
        # Fallback if not defined (e.g. if not using config override)
        assets_root = os.environ.get("LIBERO_ASSET_ROOT", "")
        if not assets_root:
             # Try to guess from imported module
             import libero.libero
             assets_root = os.path.join(os.path.dirname(libero.libero.__file__), "assets")

    print(f"[INFO] Scanning for unregistered objects in: {assets_root}")

    # Explicitly register missing articulated / special objects that require specific logic
    if (
        "yellow_cabinet" not in objects.OBJECTS_DICT
        or not hasattr(objects.OBJECTS_DICT["yellow_cabinet"], "is_close")
    ):
        print("[INFO] Manually registering YellowCabinet")
        @register_object
        class YellowCabinet(ArticulatedObject):
            def __init__(
                self,
                name="yellow_cabinet",
                obj_name="yellow_cabinet",
                joints=[dict(type="free", damping="0.0005")],
            ):
                # [FIX] Use absolute path to the asset file
                if 'project_libero_root' in globals() or 'project_libero_root' in locals():
                     # If we defined a custom root earlier
                     root = locals().get('project_libero_root', globals().get('project_libero_root'))
                     xml_path = os.path.join(root, "assets/articulated_objects/yellow_cabinet.xml")
                else:
                     # Fallback to standard resolution
                     xml_path = "articulated_objects/yellow_cabinet.xml"
                     
                # If the path is absolute, MujocoXMLObject treats it as such.
                # If it's relative, it tries to find it relative to assets dir, BUT
                # ArticulatedObject's __init__ might be hardcoding a path construction.
                # Let's inspect ArticulatedObject's __init__ again?
                # Instead of calling super().__init__ which might enforce a broken path,
                # we can call MujocoXMLObject directly if needed, OR we override the logic.
                
                # Actually, ArticulatedObject.__init__ does:
                # super().__init__(os.path.join(str(absolute_path), f"assets/articulated_objects/{obj_name}.xml"), ...)
                # This 'absolute_path' inside articulated_objects.py is relative to the INSTALLED library,
                # NOT our local project folder. This is why it fails!
                
                # Solution: We must construct the path manually and call MujocoXMLObject directly,
                # bypassing ArticulatedObject's path construction but keeping its logic.
                
                from robosuite.models.objects import MujocoXMLObject
                import numpy as np
                
                # Construct correct path
                # We know assets_root is correct (e.g. .../libero/assets)
                correct_xml_path = os.path.join(assets_root, "articulated_objects/yellow_cabinet.xml")
                
                MujocoXMLObject.__init__(
                    self,
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                
                # Copied from ArticulatedObject.__init__
                self.category_name = "yellow_cabinet"
                self.rotation = (np.pi / 4, np.pi / 2)
                self.rotation_axis = "x"
                articulation_object_properties = {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

                self.object_properties["articulation"]["default_open_ranges"] = [-0.16, -0.14]
                self.object_properties["articulation"]["default_close_ranges"] = [0.0, 0.005]

            def is_open(self, qpos):
                if qpos < max(self.object_properties["articulation"]["default_open_ranges"]):
                    return True
                else:
                    return False

            def is_close(self, qpos):
                if qpos > min(self.object_properties["articulation"]["default_close_ranges"]):
                    return True
                else:
                    return False

    if (
        "yellow_stove" not in objects.OBJECTS_DICT
        or not hasattr(objects.OBJECTS_DICT["yellow_stove"], "turn_on")
    ):
        print("[INFO] Manually registering YellowStove")
        @register_object
        @register_visual_change_object
        class YellowStove(ArticulatedObject):
            def __init__(
                self,
                name="yellow_stove",
                obj_name="yellow_stove",
                joints=[dict(type="free", damping="0.0005")],
            ):
                # Same fix for YellowStove
                from robosuite.models.objects import MujocoXMLObject
                import numpy as np
                
                correct_xml_path = os.path.join(assets_root, "articulated_objects/yellow_stove.xml")
                
                MujocoXMLObject.__init__(
                    self,
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                
                self.category_name = "yellow_stove"
                self.rotation = (np.pi / 4, np.pi / 2)
                self.rotation_axis = "x"
                articulation_object_properties = {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }
                
                self.rotation = (0, 0)
                self.rotation_axis = "y"

                tracking_sites_dict = {}
                tracking_sites_dict["burner"] = (self.naming_prefix + "burner", False)
                self.object_properties["vis_site_names"].update(tracking_sites_dict)
                self.object_properties["articulation"]["default_turnon_ranges"] = [0.5, 2.1]
                self.object_properties["articulation"]["default_turnoff_ranges"] = [-0.005, 0.0]

            def turn_on(self, qpos):
                if qpos >= min(self.object_properties["articulation"]["default_turnon_ranges"]):
                    # TODO: Set visualization sites to be true
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        True,
                    )
                    return True
                else:
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        False,
                    )
                    return False

            def turn_off(self, qpos):
                if qpos < max(self.object_properties["articulation"]["default_turnoff_ranges"]):
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        False,
                    )
                    return True
                else:
                    self.object_properties["vis_site_names"]["burner"] = (
                        self.naming_prefix + "burner",
                        True,
                    )
                    return False

    # Manually register mugs from local turbosquid assets to ensure availability
    if "porcelain_mug" not in objects.OBJECTS_DICT:
        print("[INFO] Manually registering PorcelainMug")

        @register_object
        class PorcelainMug(MujocoXMLObject):
            def __init__(
                self,
                name="porcelain_mug",
                obj_name="porcelain_mug",
                joints=[dict(type="free", damping="0.0005")],
            ):
                correct_xml_path = os.path.join(
                    assets_root, "turbosquid_objects/porcelain_mug/porcelain_mug.xml"
                )
                super().__init__(
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                self.category_name = "porcelain_mug"

                # Default tabletop rotation settings
                self.rotation = (0.0, 0.0)
                self.rotation_axis = "z"

                # Minimal object_properties to satisfy LIBERO expectations
                articulation_object_properties = {
                    "default_open_ranges": [0.0, 0.0],
                    "default_close_ranges": [0.0, 0.0],
                    "default_turnon_ranges": [0.0, 0.0],
                    "default_turnoff_ranges": [0.0, 0.0],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

    if "white_porcelain_mug" not in objects.OBJECTS_DICT:
        print("[INFO] Manually registering WhitePorcelainMug")

        @register_object
        class WhitePorcelainMug(MujocoXMLObject):
            def __init__(
                self,
                name="white_porcelain_mug",
                obj_name="white_porcelain_mug",
                joints=[dict(type="free", damping="0.0005")],
            ):
                correct_xml_path = os.path.join(
                    assets_root,
                    "turbosquid_objects/white_porcelain_mug/white_porcelain_mug.xml",
                )
                super().__init__(
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )
                self.category_name = "white_porcelain_mug"

                self.rotation = (0.0, 0.0)
                self.rotation_axis = "z"

                articulation_object_properties = {
                    "default_open_ranges": [0.0, 0.0],
                    "default_close_ranges": [0.0, 0.0],
                    "default_turnon_ranges": [0.0, 0.0],
                    "default_turnoff_ranges": [0.0, 0.0],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

    if (
        "wooden_cabinet" not in objects.OBJECTS_DICT
        or not hasattr(objects.OBJECTS_DICT["wooden_cabinet"], "is_close")
    ):
        print("[INFO] Manually registering WoodenCabinet with articulated behavior")

        @register_object
        class WoodenCabinet(ArticulatedObject):
            def __init__(
                self,
                name="wooden_cabinet",
                obj_name="wooden_cabinet",
                joints=[dict(type="free", damping="0.0005")],
            ):
                from robosuite.models.objects import MujocoXMLObject

                correct_xml_path = os.path.join(
                    assets_root, "articulated_objects/wooden_cabinet.xml"
                )

                MujocoXMLObject.__init__(
                    self,
                    correct_xml_path,
                    name=name,
                    joints=joints,
                    obj_type="all",
                    duplicate_collision_geoms=False,
                )

                self.category_name = "wooden_cabinet"
                self.rotation = (np.pi / 4, np.pi / 2)
                self.rotation_axis = "x"
                articulation_object_properties = {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                }
                self.object_properties = {
                    "articulation": articulation_object_properties,
                    "vis_site_names": {},
                }

                self.object_properties["articulation"]["default_open_ranges"] = [
                    -0.16,
                    -0.14,
                ]
                self.object_properties["articulation"]["default_close_ranges"] = [
                    0.0,
                    0.005,
                ]

            def is_open(self, qpos):
                if qpos < max(self.object_properties["articulation"]["default_open_ranges"]):
                    return True
                else:
                    return False

            def is_close(self, qpos):
                if qpos > min(self.object_properties["articulation"]["default_close_ranges"]):
                    return True
                else:
                    return False
    
    # Helper to register objects
    def register_from_folder(folder_name, base_class):
        # Explicitly reference the re module from outer scope
        import re as re_module
        search_path = os.path.join(assets_root, folder_name)
        if not os.path.exists(search_path):
            print(f"[WARN] Asset folder not found: {search_path}")
            return

        count = 0
        for obj_name in os.listdir(search_path):
            full_path = os.path.join(search_path, obj_name)
            
            # Strategy 1: obj_name is a directory containing obj_name.xml
            # Example: assets/stable_scanned_objects/apple/apple.xml
            if os.path.isdir(full_path):
                xml_path = os.path.join(full_path, f"{obj_name}.xml")
                if os.path.exists(xml_path):
                    # Register logic (same as before)
                    if obj_name.lower() not in objects.OBJECTS_DICT:
                        class_name = "".join(x.title() for x in obj_name.split("_"))
                        def make_init(x_path, o_name):
                            def __init__(self, name=o_name, obj_name=o_name, joints=[dict(type="free", damping="0.0005")]):
                                from robosuite.models.objects import MujocoXMLObject
                                import numpy as np
                                MujocoXMLObject.__init__(self, x_path, name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
                                self.category_name = o_name 
                                self.rotation = (np.pi / 2, np.pi / 2)
                                self.rotation_axis = "x"
                                self.object_properties = {"vis_site_names": {}}
                            return __init__
                        new_class = type(class_name, (MujocoXMLObject,), {"__init__": make_init(xml_path, obj_name)})
                        register_object(new_class)
                        count += 1
            
            # Strategy 2: obj_name is an XML file directly
            # Example: assets/articulated_objects/yellow_cabinet.xml
            elif obj_name.endswith(".xml"):
                # obj_name is "yellow_cabinet.xml", we want "yellow_cabinet"
                pure_name = os.path.splitext(obj_name)[0]
                
                # SKIP if it's already manually registered above (prevents overwriting our custom classes)
                if pure_name in [
                    "yellow_cabinet",
                    "yellow_stove",
                    "wooden_cabinet",
                    "porcelain_mug",
                    "white_porcelain_mug",
                ]:
                    continue

                xml_path = full_path
                
                # Force override or register
                class_name = "".join(x.title() for x in pure_name.split("_"))
                def make_init(x_path, o_name):
                    def __init__(self, name=o_name, obj_name=o_name, joints=[dict(type="free", damping="0.0005")]):
                        from robosuite.models.objects import MujocoXMLObject
                        import numpy as np
                        MujocoXMLObject.__init__(self, x_path, name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
                        self.category_name = o_name 
                        
                        # Initialize rotation (default x-axis for tabletop objects)
                        self.rotation = (np.pi / 2, np.pi / 2)
                        self.rotation_axis = "x"
                        
                        # Initialize object properties properly
                        # 'articulation' key is needed for articulated objects (e.g., microwave, stove, cabinet)
                        # 'vis_site_names' is needed for all objects
                        # Set default ranges to [0.0, 0.0] instead of [] to prevent IndexError in OpenCloseSampler
                        # If the object is actually articulated, these will be overridden by the specific class
                        articulation_object_properties = {
                            "default_open_ranges": [0.0, 0.0],
                            "default_close_ranges": [0.0, 0.0],
                            "default_turnon_ranges": [0.0, 0.0],
                            "default_turnoff_ranges": [0.0, 0.0],
                        }
                        self.object_properties = {
                            "articulation": articulation_object_properties,
                            "vis_site_names": {},
                        }
                    return __init__
                new_class = type(class_name, (MujocoXMLObject,), {"__init__": make_init(xml_path, pure_name)})
                
                # Manually register to bypass assert check in register_object.
                # Skip if a class with this key already exists (preserve articulated behaviors).
                key = "_".join(re_module.sub(r"([A-Z0-9])", r" \1", new_class.__name__).split()).lower()
                if key not in objects.OBJECTS_DICT:
                    objects.OBJECTS_DICT[key] = new_class
                    count += 1

        print(f"[INFO] Registered {count} new objects from {folder_name}")

    # Scan stable_scanned_objects and stable_hope_objects
    # Note: We use MujocoXMLObject as the base for dynamic registration, mimicking the notebook's CustomObjects
    register_from_folder("stable_scanned_objects", MujocoXMLObject)
    register_from_folder("stable_hope_objects", MujocoXMLObject)
    # Turbosquid objects structure might be similar
    register_from_folder("turbosquid_objects", MujocoXMLObject)

    # Also register objects from articulated_objects
    # articulated_objects can be complex but many are just XML models.
    # We attempt to register them similarly.
    register_from_folder("articulated_objects", MujocoXMLObject)

    # Print all registered objects
    print("\n" + "="*80)
    print("[INFO] All registered objects in OBJECTS_DICT:")
    print("="*80)
    registered_objects = sorted(objects.OBJECTS_DICT.keys())
    for i, obj_name in enumerate(registered_objects, 1):
        obj_class = objects.OBJECTS_DICT[obj_name]
        print(f"  {i:3d}. {obj_name:40s} -> {obj_class.__name__}")
    print(f"\n[INFO] Total: {len(registered_objects)} registered objects")
    print("="*80 + "\n")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[original_task_suite_name]()
    
    # [Critical Fix] Force update the BDDL file paths if we are using a perturbed suite
    if cfg.task_suite_name != original_task_suite_name:
        print(f"[INFO] Overriding BDDL paths to use perturbed suite: {cfg.task_suite_name}")
        
        # Determine the new folder name (e.g., "libero_spatial_temp" or "libero_spatial_env")
        new_folder_name = cfg.task_suite_name
        
        # We need to monkey-patch the tasks within the suite
        from libero.libero.benchmark import Task, grab_language_from_filename
        import re

        def extract_language_from_bddl(bddl_path):
            try:
                with open(bddl_path, 'r') as f:
                    content = f.read()
                match = re.search(r"\(:language\s*(.*?)\)", content, re.DOTALL)
                if match:
                    return match.group(1).strip()
            except Exception as e:
                print(f"[WARNING] Failed to extract language from {bddl_path}: {e}")
            return None

        new_tasks = []
        for old_task in task_suite.tasks:
            # Construct the full path to the NEW bddl file
            # BDDL files are in: [benchmark_root]/bddl_files/[problem_folder]/[bddl_file]
            # We need to construct this path using our temp config settings
            
            # Assuming our temp config set bddl_files to: project_libero_root/bddl_files
            # and new_folder_name is e.g. "libero_spatial_temp"
            
            project_libero_root = str(libero_pro_libero_root())
            new_bddl_full_path = os.path.join(project_libero_root, "bddl_files", new_folder_name, old_task.bddl_file)
            
            # Extract new language
            new_language = extract_language_from_bddl(new_bddl_full_path)
            if new_language:
                print(f"[INFO] Updated language for {old_task.name}: {new_language}")
            else:
                new_language = old_task.language
                print(f"[WARN] Could not update language for {old_task.name}, keeping original.")

            # Construct new path structure based on how your perturbation.py creates folders
            # Assuming perturbation creates a parallel folder structure
            new_task = Task(
                name=old_task.name,
                language=new_language,  # <--- Updated language!
                problem=old_task.problem,
                problem_folder=new_folder_name,  # <--- Point to the new folder
                bddl_file=old_task.bddl_file,
                init_states_file=old_task.init_states_file 
            )
            new_tasks.append(new_task)
        
        # Replace the tasks list in the suite
        task_suite.tasks = new_tasks

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}-{cfg.task_start}-{cfg.task_end}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # If attention saving is requested, make sure directory is set up
    date_stamp = os.getenv('DATE_STAMP', None)
    run_stamp = os.getenv('RUN_STAMP', None)
    if cfg.save_attentions:
        if cfg.save_attn_dir:
            cfg.save_attn_dir = os.path.join(cfg.save_attn_dir, cfg.task_suite_name, date_stamp, run_stamp)
        else:
            cfg.save_attn_dir = os.path.join('saved_attention/openvla', cfg.pretrained_checkpoint.split('/')[-1], cfg.task_suite_name, date_stamp, run_stamp)
        os.makedirs(cfg.save_attn_dir, exist_ok=True)
        log_file.write(
            f"[eval_libero] Attention saving ENABLED | save_attn_dir={cfg.save_attn_dir}"
        )
    else:
        log_file.write("[eval_libero] Attention saving DISABLED\n")

    self_eval_records: list = []
    self_eval_json_path = ""
    results_output_dir = ""
    if cfg.enable_self_eval:
        results_output_dir = resolve_self_eval_output_dir(cfg, date_stamp=date_stamp, run_stamp=run_stamp)
        os.makedirs(results_output_dir, exist_ok=True)
        self_eval_json_path = os.path.join(
            results_output_dir,
            self_eval_json_filename(
                cfg.self_eval_mode,
                attention_eval_output_name=cfg.attention_eval_output_name,
            ),
        )
        log_file.write(
            f"\n[eval_libero] Self-eval ENABLED | mode={cfg.self_eval_mode} | json={self_eval_json_path}\n"
        )
        if cfg.self_eval_mode == MAE_D_MODE:
            log_file.write(
                f"[eval_libero] Online MAE-D eval | ratios={cfg.attention_eval_ratios_list} "
                f"| no attention save\n"
            )
        elif cfg.self_eval_mode == "consistency":
            log_file.write(
                f"[eval_libero] Consistency method: {getattr(cfg, 'consistency_method', 'token_sample')}\n"
            )
            if str(getattr(cfg, "consistency_method", "token_sample")).strip().lower() == "token_sample":
                log_file.write(
                    f"[eval_libero] Token samples per action step: {cfg.consistency_repeats_per_init}\n"
                )
            else:
                log_file.write(
                    f"[eval_libero] Consistency repeats per init: {cfg.consistency_repeats_per_init}\n"
                )
            log_file.write(
                f"[eval_libero] Action sampling temperature: {cfg.action_sampling_temperature}\n"
            )
    else:
        log_file.write("\n[eval_libero] Self-eval DISABLED\n")

    output_stats_jsonl_path = ""
    if cfg.track_output_stats and not cfg.enable_self_eval:
        stats_root = cfg.save_attn_dir or resolve_self_eval_output_dir(cfg, date_stamp=date_stamp, run_stamp=run_stamp)
        os.makedirs(stats_root, exist_ok=True)
        output_stats_jsonl_path = os.path.join(stats_root, "online_output_stats.jsonl")
        log_file.write(
            f"\n[eval_libero] Output stats ENABLED | jsonl={output_stats_jsonl_path}\n"
        )
    elif cfg.track_output_stats:
        log_file.write("\n[eval_libero] Output stats routed to self-eval JSON\n")
    else:
        log_file.write("\n[eval_libero] Output stats DISABLED\n")


    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Determine task range
    task_start = cfg.task_start if cfg.task_start >= 0 else 0
    task_end = cfg.task_end if cfg.task_end >= 0 else num_tasks_in_suite
    
    cfg.task_start = task_start
    cfg.task_end = task_end

    # Validate task range
    if task_start < 0 or task_start >= num_tasks_in_suite:
        raise ValueError(f"task_start ({task_start}) must be in range [0, {num_tasks_in_suite})")
    if task_end <= task_start or task_end > num_tasks_in_suite:
        raise ValueError(f"task_end ({task_end}) must be in range ({task_start}, {num_tasks_in_suite}]")

    tasks_to_evaluate = list(range(task_start, task_end))
    num_tasks_to_evaluate = len(tasks_to_evaluate)


    print(f"Total tasks in suite: {num_tasks_in_suite}")
    print(f"Evaluating tasks: {task_start} to {task_end-1} (inclusive, {num_tasks_to_evaluate} tasks)")
    print(f"benchmark_dict: {benchmark_dict}")
    print(f"task_suite: {task_suite}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)



    # Start evaluation
    total_episodes, total_successes = 0, 0
    final_results = {'success': [], 'failure': []}
    for task_id in tqdm.tqdm(tasks_to_evaluate):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            use_token_sample_consistency = (
                cfg.enable_self_eval
                and cfg.self_eval_mode == "consistency"
                and str(getattr(cfg, "consistency_method", "token_sample")).strip().lower() == "token_sample"
            )
            num_repeats = (
                1
                if use_token_sample_consistency
                else (
                    cfg.consistency_repeats_per_init
                    if cfg.enable_self_eval and cfg.self_eval_mode == "consistency"
                    else 1
                )
            )
            repeat_successes = []
            episode_token_step_scores = []

            for repeat_idx in range(num_repeats):
                if cfg.action_sampling_temperature > 0 and not use_token_sample_consistency:
                    set_seed_everywhere(cfg.seed + task_id * 10000 + episode_idx * 100 + repeat_idx)

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                replay_images = []
                suite_name = cfg.task_suite_name
                if suite_name in {
                    "libero_spatial", "libero_spatial_temp_swap",
                    "libero_object", "libero_object_temp_swap",
                    "libero_goal", "libero_goal_temp_swap",
                    "libero_10", "libero_10_temp_swap",
                }:
                    max_steps = 520
                elif suite_name == "libero_90":
                    max_steps = 400
                else:
                    max_steps = 520

                print(
                    f"Starting episode {task_episodes + 1} "
                    f"(init {episode_idx + 1}, repeat {repeat_idx + 1}/{num_repeats})..."
                )
                log_file.write(
                    f"Starting episode {task_episodes + 1} "
                    f"(init {episode_idx + 1}, repeat {repeat_idx + 1}/{num_repeats})...\n"
                )
                query_idx = 0
                episode_query_output_stats = []
                episode_visual_entropy_queries = []
                done = False
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        img = get_libero_image(obs, resize_size)
                        replay_images.append(img)

                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }
                        if rollout_requests_attentions(cfg) or cfg.track_output_stats:
                            result = get_action(
                                cfg,
                                model,
                                observation,
                                task_description,
                                processor=processor,
                            )
                            output_stats = None
                            attentions = None
                            processed_query = False
                            if rollout_requests_attentions(cfg) and cfg.track_output_stats:
                                action, attentions, num_patches, num_prompt_tokens, output_stats = result
                            elif rollout_requests_attentions(cfg):
                                action, attentions, num_patches, num_prompt_tokens = result
                            else:
                                action, output_stats = result

                            if output_stats is not None:
                                output_stats = dict(output_stats)
                                output_stats["query_idx"] = int(query_idx)
                                episode_query_output_stats.append(output_stats)
                                processed_query = True

                            if attentions is not None:
                                processed_query = True
                                if isinstance(attentions, tuple):
                                    attentions_cpu = tuple(
                                        layer_attn.cpu() if hasattr(layer_attn, "cpu") else layer_attn
                                        for layer_attn in attentions
                                    )
                                else:
                                    attentions_cpu = (
                                        attentions.cpu() if hasattr(attentions, "cpu") else attentions
                                    )
                                del attentions

                                if cfg.enable_self_eval and cfg.self_eval_mode == MAE_D_MODE:
                                    vis_lh = compute_visual_entropy_lh_from_attentions(
                                        attentions_cpu,
                                        num_patches,
                                        num_prompt_tokens,
                                        variant="openvla",
                                    )
                                    if vis_lh is not None:
                                        episode_visual_entropy_queries.append(vis_lh)
                                elif cfg.save_attentions and cfg.save_attn_dir and task_id >= 0 and episode_idx >= 0:
                                    save_single_attention_weight(
                                        cfg,
                                        attentions_cpu,
                                        task_id,
                                        episode_idx,
                                        query_idx,
                                        task_description,
                                        img,
                                        False,
                                        total_episodes + 1,
                                        log_file,
                                        num_patches=num_patches,
                                        num_prompt_tokens=num_prompt_tokens,
                                    )
                                del attentions_cpu

                            if processed_query:
                                query_idx += 1
                                if query_idx % 10 == 0:
                                    torch.cuda.empty_cache()
                        elif use_token_sample_consistency:
                            num_token_samples = cfg.consistency_repeats_per_init
                            token_samples = []
                            action = None
                            for sample_idx in range(num_token_samples):
                                if cfg.action_sampling_temperature > 0:
                                    set_seed_everywhere(
                                        cfg.seed
                                        + task_id * 100000
                                        + episode_idx * 1000
                                        + t * num_token_samples
                                        + sample_idx
                                    )
                                sample_action, token_ids = get_action_with_token_ids(
                                    cfg,
                                    model,
                                    observation,
                                    task_description,
                                    processor=processor,
                                )
                                token_samples.append(token_ids)
                                if sample_idx == 0:
                                    action = sample_action
                            step_score = compute_step_token_consistency(token_samples)
                            if step_score is not None:
                                episode_token_step_scores.append(step_score)

                        else:
                            action = get_action(
                                cfg,
                                model,
                                observation,
                                task_description,
                                processor=processor,
                            )

                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            break
                        t += 1

                    except Exception as e:
                        print(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        break

                task_episodes += 1
                total_episodes += 1
                repeat_successes.append(bool(done))
                if done:
                    task_successes += 1
                    total_successes += 1

                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    run_stamp=run_stamp,
                    log_file=log_file,
                )

                if (
                    cfg.enable_self_eval
                    and cfg.self_eval_mode == MAE_D_MODE
                    and episode_visual_entropy_queries
                ):
                    mad_record = build_episode_mad_record(
                        task_id=task_id + 1,
                        task_description=task_description,
                        episode_idx=episode_idx + 1,
                        repeat_idx=repeat_idx + 1,
                        episode_number=total_episodes,
                        is_success=bool(done),
                        episode_visual_entropy_queries=episode_visual_entropy_queries,
                        eval_ratios=cfg.attention_eval_ratios_list,
                    )
                    if mad_record is not None:
                        append_online_mad_jsonl(mad_record, self_eval_json_path)
                elif cfg.enable_self_eval and cfg.self_eval_mode == "output_stats" and episode_query_output_stats:
                    episode_stats = aggregate_episode_output_stats(episode_query_output_stats)
                    if episode_stats is not None:
                        record = {
                            "task_id": int(task_id + 1),
                            "task_description": task_description,
                            "episode_idx": int(episode_idx + 1),
                            "repeat_idx": int(repeat_idx + 1),
                            "episode_number": int(total_episodes),
                            "is_success": bool(done),
                            "num_queries_used": int(len(episode_query_output_stats)),
                            **episode_stats,
                        }
                        self_eval_records.append(record)
                        write_self_eval_json(
                            {
                                "self_eval_mode": cfg.self_eval_mode,
                                "libero_benchmark": cfg.libero_benchmark,
                                "task_suite_name": cfg.task_suite_name,
                                "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
                                "records": self_eval_records,
                            },
                            self_eval_json_path,
                        )
                elif cfg.track_output_stats and episode_query_output_stats and not cfg.enable_self_eval:
                    episode_stats = aggregate_episode_output_stats(episode_query_output_stats)
                    if episode_stats is not None:
                        from experiments.robot.openvla_utils import append_output_stats_jsonl

                        append_output_stats_jsonl(
                            {
                                "task_id": int(task_id + 1),
                                "task_description": task_description,
                                "episode_idx": int(episode_idx + 1),
                                "episode_number": int(total_episodes),
                                "is_success": bool(done),
                                "num_queries_used": int(len(episode_query_output_stats)),
                                **episode_stats,
                            },
                            output_stats_jsonl_path,
                        )

                if cfg.enable_self_eval and cfg.self_eval_mode != "consistency":
                    if done:
                        final_results["success"].append((task_id, episode_idx))
                    else:
                        final_results["failure"].append((task_id, episode_idx))

                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(
                    f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
                )
                log_file.flush()

            if cfg.enable_self_eval and cfg.self_eval_mode == "consistency":
                if use_token_sample_consistency:
                    consistency_score = aggregate_episode_token_consistency(episode_token_step_scores)
                    record = {
                        "task_id": int(task_id + 1),
                        "task_description": task_description,
                        "episode_idx": int(episode_idx + 1),
                        "consistency_method": "token_sample",
                        "num_token_samples_per_step": int(cfg.consistency_repeats_per_init),
                        "num_action_steps_scored": int(len(episode_token_step_scores)),
                        "consistency_score": consistency_score,
                        "is_success": bool(repeat_successes[0]) if repeat_successes else False,
                    }
                    if consistency_score is not None and repeat_successes:
                        final_results["success" if repeat_successes[0] else "failure"].append(
                            (task_id, episode_idx)
                        )
                else:
                    success_count = int(sum(repeat_successes))
                    consistency_score = float(success_count / num_repeats)
                    record = {
                        "task_id": int(task_id + 1),
                        "task_description": task_description,
                        "episode_idx": int(episode_idx + 1),
                        "consistency_method": "outcome_repeat",
                        "num_repeats": int(num_repeats),
                        "success_count": success_count,
                        "consistency_score": consistency_score,
                        "repeat_successes": repeat_successes,
                    }
                    if consistency_score > 0:
                        final_results["success"].append((task_id, episode_idx))
                    else:
                        final_results["failure"].append((task_id, episode_idx))
                self_eval_records.append(record)
                write_self_eval_json(
                    {
                        "self_eval_mode": cfg.self_eval_mode,
                        "consistency_method": str(getattr(cfg, "consistency_method", "token_sample")),
                        "libero_benchmark": cfg.libero_benchmark,
                        "task_suite_name": cfg.task_suite_name,
                        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
                        "consistency_repeats_per_init": int(cfg.consistency_repeats_per_init),
                        "action_sampling_temperature": float(cfg.action_sampling_temperature),
                        "records": self_eval_records,
                    },
                    self_eval_json_path,
                )

            save_result_path = os.path.join(
                results_output_dir or cfg.save_attn_dir or ".",
                f"final_results{cfg.task_start}-{cfg.task_end}.yaml",
            )
            if (episode_idx + 1) % 10 == 0:
                os.makedirs(os.path.dirname(save_result_path) or ".", exist_ok=True)
                with open(save_result_path, "w", encoding="utf-8") as f:
                    yaml.dump(final_results, f, allow_unicode=True)
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    # debugpy.listen(("127.0.0.1", 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    eval_libero()
