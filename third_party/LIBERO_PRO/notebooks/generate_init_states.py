#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_init_states.py
- 从指定 bddl_base_dir 读取所有 .bddl
- 为每个任务生成 num_inits 个初始状态
- 保存为 .pruned_init 压缩文件到 output_dir

用法示例：
    python generate_init_states.py \
        --bddl_base_dir /path/to/bddl_dir \
        --output_dir /path/to/output_dir \
        --num_inits 50 \
        --height 128 \
        --width 128
"""

import os
import re
import zipfile
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import traceback

# [CRITICAL FIX] Register missing objects from assets
# This script runs as a subprocess, so it needs its own object registration logic.
from libero.libero.envs.base_object import register_object, register_visual_change_object
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs import objects
from libero.libero.envs.objects.articulated_objects import ArticulatedObject
# Force import of turbosquid_objects so built-in PorcelainMug, WhiteYellowMug, etc. are registered
from libero.libero.envs.objects import turbosquid_objects  # noqa: F401
import libero.libero

# Try to find assets directory
project_libero_root = str(Path(__file__).resolve().parents[1] / "libero" / "libero")
assets_root = os.path.join(project_libero_root, "assets")
if not os.path.exists(assets_root):
    # Fallback
    assets_root = os.path.join(os.path.dirname(libero.libero.__file__), "assets")

print(f"[INFO] Scanning for unregistered objects in: {assets_root}")

# Explicitly register missing articulated objects / special objects that require specific logic
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
            from robosuite.models.objects import MujocoXMLObject
            import numpy as np
            # Construct correct path using assets_root
            correct_xml_path = os.path.join(assets_root, "articulated_objects/yellow_cabinet.xml")
            MujocoXMLObject.__init__(
                self,
                correct_xml_path,
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
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

# Save reference to WhitePorcelainMug class for re-registration if needed
_WhitePorcelainMugClass = None

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
    
    _WhitePorcelainMugClass = WhitePorcelainMug
else:
    # If already registered, save reference for potential re-registration
    _WhitePorcelainMugClass = objects.OBJECTS_DICT["white_porcelain_mug"]

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

def register_from_folder(folder_name, base_class):
    search_path = os.path.join(assets_root, folder_name)
    if not os.path.exists(search_path):
        print(f"[WARN] Asset folder not found: {search_path}")
        return

    count = 0
    for obj_name in os.listdir(search_path):
        full_path = os.path.join(search_path, obj_name)
        
        # Strategy 1: obj_name is a directory containing obj_name.xml
        if os.path.isdir(full_path):
            xml_path = os.path.join(full_path, f"{obj_name}.xml")
            if os.path.exists(xml_path):
                if obj_name.lower() not in objects.OBJECTS_DICT:
                    class_name = "".join(x.title() for x in obj_name.split("_"))
                    def make_init(x_path, o_name):
                        def __init__(self, name=o_name, obj_name=o_name, joints=[dict(type="free", damping="0.0005")]):
                            MujocoXMLObject.__init__(self, x_path, name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
                            self.category_name = o_name
                            self.object_properties = {"vis_site_names": {}}
                            self.rotation = (np.pi / 2, np.pi / 2)
                            self.rotation_axis = "x"
                        return __init__

                    new_class = type(class_name, (MujocoXMLObject,), {"__init__": make_init(xml_path, obj_name)})
                    register_object(new_class)
                    count += 1
        
        # Strategy 2: obj_name is an XML file directly
        elif obj_name.endswith(".xml"):
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
            
            # Manually register instead of using @register_object to bypass the assert check if it exists
            key = "_".join(re.sub(r"([A-Z0-9])", r" \1", new_class.__name__).split()).lower()
            if key not in objects.OBJECTS_DICT:
                objects.OBJECTS_DICT[key] = new_class
                count += 1

    print(f"[INFO] Registered {count} new objects from {folder_name}")

# Register objects
register_from_folder("stable_scanned_objects", MujocoXMLObject)
register_from_folder("stable_hope_objects", MujocoXMLObject)
register_from_folder("turbosquid_objects", MujocoXMLObject)

# Also register objects from articulated_objects
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

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.objects import get_object_fn as original_get_object_fn

# After importing OffScreenRenderEnv, re-register white_porcelain_mug if needed
# (imports might have reset OBJECTS_DICT)
def ensure_white_porcelain_mug_registered():
    """Ensure white_porcelain_mug is registered in OBJECTS_DICT"""
    if "white_porcelain_mug" not in objects.OBJECTS_DICT:
        if _WhitePorcelainMugClass is not None:
            objects.OBJECTS_DICT["white_porcelain_mug"] = _WhitePorcelainMugClass
            return True
        else:
            print("[ERROR] white_porcelain_mug not in OBJECTS_DICT and _WhitePorcelainMugClass is None!")
            return False
    return True

# Create a wrapper for get_object_fn with fallback logic
def get_object_fn_with_fallback(category_name):
    """Wrapper for get_object_fn that falls back to porcelain_mug if white_porcelain_mug is not found"""
    category_name_lower = category_name.lower()
    
    # First, ensure white_porcelain_mug is registered if needed
    if category_name_lower == "white_porcelain_mug" and "white_porcelain_mug" not in objects.OBJECTS_DICT:
        if _WhitePorcelainMugClass is not None:
            objects.OBJECTS_DICT["white_porcelain_mug"] = _WhitePorcelainMugClass
            print(f"[DEBUG] Re-registered white_porcelain_mug in get_object_fn_with_fallback")
        else:
            print(f"[WARN] white_porcelain_mug not found and _WhitePorcelainMugClass is None, falling back to porcelain_mug")
            try:
                return original_get_object_fn("porcelain_mug")
            except KeyError:
                raise KeyError(f"Neither white_porcelain_mug nor porcelain_mug found in OBJECTS_DICT")
    
    try:
        return original_get_object_fn(category_name)
    except KeyError:
        if category_name_lower == "white_porcelain_mug":
            # Fallback to porcelain_mug if white_porcelain_mug is not found
            print(f"[WARN] white_porcelain_mug not found in OBJECTS_DICT, falling back to porcelain_mug")
            try:
                return original_get_object_fn("porcelain_mug")
            except KeyError:
                # If porcelain_mug is also not found, try to use our registered class
                if _WhitePorcelainMugClass is not None:
                    objects.OBJECTS_DICT["white_porcelain_mug"] = _WhitePorcelainMugClass
                    return _WhitePorcelainMugClass
                raise KeyError(f"Neither white_porcelain_mug nor porcelain_mug found in OBJECTS_DICT")
        raise

# Monkey patch get_object_fn to use our fallback version
import libero.libero.envs.objects
libero.libero.envs.objects.get_object_fn = get_object_fn_with_fallback

# Monkey patch _load_objects_in_arena to add exception handling and debugging
# We'll do this dynamically when the class is actually used
def _load_objects_in_arena_with_debug(self, mujoco_arena):
    """Wrapper for _load_objects_in_arena with exception handling and debugging"""
    print(f"[DEBUG] _load_objects_in_arena_with_debug called!")
    objects_dict = self.parsed_problem["objects"]
    print(f"[DEBUG] Objects to create from parsed_problem: {objects_dict}")
    failed_objects = []
    
    for category_name in objects_dict.keys():
        for object_name in objects_dict[category_name]:
            print(f"[DEBUG] Attempting to create {object_name} (type: {category_name})")
            try:
                # Ensure white_porcelain_mug is registered before creating instance
                if category_name.lower() == "white_porcelain_mug" and "white_porcelain_mug" not in objects.OBJECTS_DICT:
                    if _WhitePorcelainMugClass is not None:
                        objects.OBJECTS_DICT["white_porcelain_mug"] = _WhitePorcelainMugClass
                        print(f"[DEBUG] Re-registered white_porcelain_mug before creating {object_name}")
                
                obj_class = get_object_fn_with_fallback(category_name)
                self.objects_dict[object_name] = obj_class(name=object_name)
                print(f"[DEBUG] Successfully created {object_name} (type: {category_name})")
            except Exception as e:
                failed_objects.append((object_name, category_name, str(e)))
                print(f"[ERROR] Failed to create {object_name} (type: {category_name}): {e}")
                import traceback
                traceback.print_exc()
                # Try fallback for white_porcelain_mug
                if category_name.lower() == "white_porcelain_mug":
                    try:
                        print(f"[DEBUG] Trying fallback: creating {object_name} as porcelain_mug")
                        obj_class = get_object_fn_with_fallback("porcelain_mug")
                        self.objects_dict[object_name] = obj_class(name=object_name)
                        print(f"[DEBUG] Successfully created {object_name} using porcelain_mug fallback")
                        failed_objects.pop()  # Remove from failed list
                    except Exception as e2:
                        print(f"[ERROR] Fallback also failed for {object_name}: {e2}")
    
    if failed_objects:
        print(f"[WARN] Failed to create {len(failed_objects)} objects: {failed_objects}")
    else:
        print(f"[DEBUG] Successfully created all objects: {list(self.objects_dict.keys())}")

# Try to monkey patch the class dynamically
try:
    from libero.libero.envs.problems.libero_living_room_tabletop_manipulation import Libero_Living_Room_Tabletop_Manipulation
    if Libero_Living_Room_Tabletop_Manipulation is not None and hasattr(Libero_Living_Room_Tabletop_Manipulation, '_load_objects_in_arena'):
        Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena = _load_objects_in_arena_with_debug
        print("[DEBUG] Successfully monkey patched Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena")
    else:
        print("[WARN] Could not monkey patch Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena (class not found or method missing)")
except Exception as e:
    print(f"[WARN] Could not import Libero_Living_Room_Tabletop_Manipulation for monkey patching: {e}")
    print("[INFO] Will try to patch it dynamically when environment is created")

# Initial check after import
if not ensure_white_porcelain_mug_registered():
    print("[WARN] Failed to register white_porcelain_mug after importing OffScreenRenderEnv")


def generate_init_states(
    bddl_base_dir: str,
    output_dir: str,
    num_inits: int = 50,
    height: int = 128,
    width: int = 128,
):
    # Try to monkey patch the class dynamically if not already done
    try:
        from libero.libero.envs.problems.libero_living_room_tabletop_manipulation import Libero_Living_Room_Tabletop_Manipulation
        if Libero_Living_Room_Tabletop_Manipulation is not None and hasattr(Libero_Living_Room_Tabletop_Manipulation, '_load_objects_in_arena'):
            if Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena != _load_objects_in_arena_with_debug:
                Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena = _load_objects_in_arena_with_debug
                print("[DEBUG] Successfully monkey patched Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena in generate_init_states")
    except Exception as e:
        print(f"[WARN] Could not monkey patch Libero_Living_Room_Tabletop_Manipulation in generate_init_states: {e}")
    
    bddl_base_dir = Path(bddl_base_dir).resolve()
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .bddl 文件
    bddl_files = list(bddl_base_dir.glob("*.bddl"))
    print(f"找到 {len(bddl_files)} 个 BDDL 文件")

    for bddl_file in tqdm(bddl_files, desc="处理 BDDL 文件"):
        task_base_name = bddl_file.stem
        print(f"\n开始处理任务: {task_base_name}")

        all_initial_states = []

        for i in tqdm(range(num_inits), desc=f"生成 {task_base_name} 的初始状态"):
            env = None
            try:
                # Ensure white_porcelain_mug is registered before creating environment
                # (LIBERO's imports might reset OBJECTS_DICT, so we need to re-register if missing)
                if not ensure_white_porcelain_mug_registered():
                    if i == 0:  # Only print once per task
                        print(f"[ERROR] Failed to register white_porcelain_mug before creating environment for {task_base_name}")
                
                # Parse BDDL to check if all required object types are registered
                if i == 0:  # Only check once per task
                    from libero.libero.envs.bddl_utils import robosuite_parse_problem
                    parsed = robosuite_parse_problem(str(bddl_file))
                    required_types = set(parsed["objects"].keys())
                    missing_types = [t for t in required_types if t.lower() not in objects.OBJECTS_DICT]
                    if missing_types:
                        print(f"[WARN] Missing object types in OBJECTS_DICT for {task_base_name}: {missing_types}")
                        # Try to register white_porcelain_mug if it's missing
                        if "white_porcelain_mug" in missing_types and _WhitePorcelainMugClass is not None:
                            objects.OBJECTS_DICT["white_porcelain_mug"] = _WhitePorcelainMugClass
                            print(f"[INFO] Registered white_porcelain_mug after BDDL parsing")
                    
                    # Debug: print all object instances that will be created
                    print(f"[DEBUG] Objects to be created for {task_base_name}:")
                    for cat_name, obj_names in parsed["objects"].items():
                        for obj_name in obj_names:
                            cat_lower = cat_name.lower()
                            if cat_lower in objects.OBJECTS_DICT:
                                print(f"  {obj_name} -> {cat_name} (registered: {objects.OBJECTS_DICT[cat_lower].__name__})")
                            else:
                                print(f"  {obj_name} -> {cat_name} (NOT REGISTERED!)")
                
                # Double-check white_porcelain_mug is registered right before creating environment
                if "white_porcelain_mug" not in objects.OBJECTS_DICT and _WhitePorcelainMugClass is not None:
                    objects.OBJECTS_DICT["white_porcelain_mug"] = _WhitePorcelainMugClass
                    if i == 0:
                        print(f"[DEBUG] Re-registered white_porcelain_mug right before creating environment")
                
                # Try to apply monkey patch dynamically before creating environment
                try:
                    from libero.libero.envs.problems.libero_living_room_tabletop_manipulation import Libero_Living_Room_Tabletop_Manipulation
                    if Libero_Living_Room_Tabletop_Manipulation is not None and hasattr(Libero_Living_Room_Tabletop_Manipulation, '_load_objects_in_arena'):
                        if Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena != _load_objects_in_arena_with_debug:
                            Libero_Living_Room_Tabletop_Manipulation._load_objects_in_arena = _load_objects_in_arena_with_debug
                            if i == 0:
                                print(f"[DEBUG] Applied monkey patch to _load_objects_in_arena before creating environment for {task_base_name}")
                except Exception as e:
                    if i == 0:
                        print(f"[WARN] Could not apply monkey patch before creating environment: {e}")
                
                env_args = {
                    "bddl_file_name": str(bddl_file),
                    "camera_heights": height,
                    "camera_widths": width,
                }
                env = OffScreenRenderEnv(**env_args)
                
                # Debug: check if all required objects were created
                if i == 0 and hasattr(env, 'env') and hasattr(env.env, 'objects_dict'):
                    from libero.libero.envs.bddl_utils import robosuite_parse_problem
                    parsed = robosuite_parse_problem(str(bddl_file))
                    required_objects = []
                    for obj_names in parsed["objects"].values():
                        required_objects.extend(obj_names)
                    
                    missing_objects = [obj_name for obj_name in required_objects if obj_name not in env.env.objects_dict]
                    if missing_objects:
                        print(f"[ERROR] Some required objects were not created in objects_dict: {missing_objects}")
                        print(f"[DEBUG] Available objects in objects_dict: {list(env.env.objects_dict.keys())}")
                    else:
                        print(f"[DEBUG] All required objects successfully created in objects_dict")
                        # Specifically check porcelain_mug_1 if it's required
                        if "porcelain_mug_1" in required_objects:
                            print(f"[DEBUG] porcelain_mug_1 successfully created in objects_dict")

                initial_state = env.get_sim_state()
                all_initial_states.append(initial_state)

            except Exception as e:
                print(f"  生成第 {i+1} 个状态时出错: {e}")
                import traceback
                traceback.print_exc()

            finally:
                if env is not None and hasattr(env, 'close'):
                    env.close()

        output_filename = f"{task_base_name}.pruned_init"
        output_filepath = output_dir / output_filename

        try:
            with zipfile.ZipFile(output_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                all_initial_states = np.array(all_initial_states)
                pickled_states_list = pickle.dumps(all_initial_states)
                zipf.writestr("archive/data.pkl", pickled_states_list)
                zipf.writestr("archive/version", b"1")

            print(f"成功保存 {len(all_initial_states)} 个状态到: {output_filepath}")

        except Exception as e:
            print(f"保存状态列表时出错: {e}")

    print("\n所有任务处理完成！")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate init states for LIBERO BDDL tasks.")
    parser.add_argument("--bddl_base_dir", type=str, required=True, help="Directory containing BDDL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pruned_init files.")
    parser.add_argument("--num_inits", type=int, default=50, help="Number of init states to generate per task.")
    parser.add_argument("--height", type=int, default=128, help="Camera height.")
    parser.add_argument("--width", type=int, default=128, help="Camera width.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_init_states(
        bddl_base_dir=args.bddl_base_dir,
        output_dir=args.output_dir,
        num_inits=args.num_inits,
        height=args.height,
        width=args.width,
    )