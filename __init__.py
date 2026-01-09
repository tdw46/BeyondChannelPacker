# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License,
# version 2 or later, as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# ***** END GPL LICENSE BLOCK *****

from __future__ import annotations

import re
from pathlib import Path

import bpy


_MULTI_SRGB_GAMMA = 1.0
_MULTI_SRGB_CONTRAST = 1.0
_MULTI_SRGB_BRIGHTNESS = 0.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_DEFAULT_SAVE_PATH_SCENES_PENDING: set[str] = set()


def _needs_default_save_paths(props) -> bool:
    if props is None:
        return False
    if (getattr(props, "save_path_single", "") or "").strip() == "" and props.rgb_image:
        return True
    if (getattr(props, "save_path_multi", "") or "").strip() == "" and props.r_image:
        return True
    if (
        getattr(props, "save_path_split", "") or ""
    ).strip() == "" and props.split_image:
        return True
    if props.split_image:
        if (
            not getattr(props, "split_name_r_is_custom", False)
            and not (getattr(props, "split_name_r", "") or "").strip()
        ):
            return True
        if (
            not getattr(props, "split_name_g_is_custom", False)
            and not (getattr(props, "split_name_g", "") or "").strip()
        ):
            return True
        if (
            not getattr(props, "split_name_b_is_custom", False)
            and not (getattr(props, "split_name_b", "") or "").strip()
        ):
            return True
        if (
            not getattr(props, "split_name_a_is_custom", False)
            and not (getattr(props, "split_name_a", "") or "").strip()
        ):
            return True
    return False


def _schedule_default_save_paths(scene) -> None:
    if scene is None:
        return
    try:
        scene_name = scene.name
    except Exception:
        return
    if not scene_name:
        return
    if scene_name in _DEFAULT_SAVE_PATH_SCENES_PENDING:
        return
    _DEFAULT_SAVE_PATH_SCENES_PENDING.add(scene_name)

    def _timer():
        try:
            scn = bpy.data.scenes.get(scene_name)
            if scn is None:
                return None
            props = getattr(scn, "beyond_channel_packer", None)
            if props is None:
                return None
            if not _needs_default_save_paths(props):
                return None
            _ensure_default_save_paths(props)
            return None
        finally:
            _DEFAULT_SAVE_PATH_SCENES_PENDING.discard(scene_name)

    try:
        bpy.app.timers.register(_timer, first_interval=0.0)
    except Exception:
        _DEFAULT_SAVE_PATH_SCENES_PENDING.discard(scene_name)


def _iter_image_editor_spaces(context):
    wm = getattr(context, "window_manager", None)
    if wm is not None:
        for window in wm.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type != "IMAGE_EDITOR":
                    continue
                space = area.spaces.active
                if space is not None:
                    yield space
        return

    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type != "IMAGE_EDITOR":
            continue
        space = area.spaces.active
        if space is not None:
            yield space


def _set_active_image_in_image_editor(context, image):
    if image is None:
        return
    for space in _iter_image_editor_spaces(context):
        space.image = image
        try:
            space.display_channels = "COLOR_ALPHA"
        except Exception:
            pass
        break


def _batch_list_index_update(items_attr: str, index_attr: str):
    def _update(self, context):
        if context is None:
            return
        try:
            items = getattr(self, items_attr)
            idx = int(getattr(self, index_attr, 0) or 0)
        except Exception:
            return
        try:
            if idx < 0 or idx >= len(items):
                return
        except Exception:
            return
        try:
            item = items[idx]
        except Exception:
            return
        img = getattr(item, "image", None)
        if img is None:
            return
        _set_active_image_in_image_editor(context, img)

    return _update


def _image_file_format_from_path(filepath: str) -> str | None:
    ext = Path(filepath).suffix.lower()
    return {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tga": "TARGA",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".bmp": "BMP",
        ".exr": "OPEN_EXR",
        ".hdr": "HDR",
    }.get(ext)


def _autosave_target_for_sources(source_images):
    for img in source_images:
        if img is None:
            continue
        filepath = (
            getattr(img, "filepath_raw", "") or getattr(img, "filepath", "") or ""
        ).strip()
        if not filepath:
            continue
        filepath = bpy.path.abspath(filepath)
        suffix = Path(filepath).suffix
        if not suffix:
            continue
        return filepath

    blend_dir = bpy.path.abspath("//")
    if blend_dir:
        return str(Path(blend_dir) / "ChannelPacked.png")
    return str(Path(bpy.app.tempdir or ".") / "ChannelPacked.png")


def _autosave_packed_target_for_sources(source_images) -> str:
    for img in source_images:
        if img is None:
            continue
        filepath = (
            getattr(img, "filepath_raw", "") or getattr(img, "filepath", "") or ""
        ).strip()
        if not filepath:
            continue
        filepath = bpy.path.abspath(filepath)
        src_path = Path(filepath)
        suffix = src_path.suffix
        if not suffix:
            continue
        return str(src_path.with_name(f"{src_path.stem}_packed{suffix}"))

    blend_dir = bpy.path.abspath("//")
    if blend_dir:
        return str(Path(blend_dir) / "ChannelPacked.png")
    return str(Path(bpy.app.tempdir or ".") / "ChannelPacked.png")


def _result_image_base_name(*, mode: str, base_stem: str) -> str:
    stem = (base_stem or "").strip() or "ChannelPacked"
    if stem != "ChannelPacked":
        return stem
    if mode == "SINGLE":
        return "ChannelPacked_RGB_A"
    if mode == "MULTI":
        return "ChannelPacked_4CH"
    return stem


def _autosave_format_for_path(filepath: str) -> tuple[str, str]:
    ext = Path(filepath).suffix.lower() if filepath else ""
    if ext == ".exr":
        return "OPEN_EXR", ".exr"
    if ext in {".tif", ".tiff"}:
        return "TIFF", ext
    if ext == ".tga":
        return "TARGA", ".tga"
    return "PNG", ".png"


def _resolve_save_target_path(
    *,
    base_path: str,
    sources,
    override_existing: bool,
) -> tuple[Path, str]:
    base = (base_path or "").strip()
    base_target = base if base else _autosave_packed_target_for_sources(sources)

    base_path = Path(bpy.path.abspath(base_target))
    file_format, ext = _autosave_format_for_path(str(base_path))

    if base_path.suffix:
        detected = _image_file_format_from_path(str(base_path))
        if detected is not None:
            file_format = detected
            ext = base_path.suffix

    if override_existing:
        out_path = base_path if base_path.suffix else base_path.with_suffix(ext)
        return out_path, file_format

    out_dir = base_path.parent if str(base_path.parent) else Path(".")
    stem = base_path.stem or "ChannelPacked"
    if stem == "ChannelPacked" or stem.endswith("_packed"):
        out_stem = stem
    else:
        out_stem = f"{stem}_packed"
    return _unique_path(out_dir / f"{out_stem}{ext}"), file_format


def _mark_save_single_custom(self, _context):
    self.save_path_single_is_custom = bool((self.save_path_single or "").strip())


def _mark_save_multi_custom(self, _context):
    self.save_path_multi_is_custom = bool((self.save_path_multi or "").strip())


def _mark_save_split_custom(self, _context):
    self.save_path_split_is_custom = bool((self.save_path_split or "").strip())


def _mark_batch_dest_single_custom(self, _context):
    self.batch_dest_dir_single_is_custom = bool(
        (getattr(self, "batch_dest_dir_single", "") or "").strip()
    )


def _mark_batch_dest_multi_custom(self, _context):
    self.batch_dest_dir_multi_is_custom = bool(
        (getattr(self, "batch_dest_dir_multi", "") or "").strip()
    )


def _mark_batch_dest_split_custom(self, _context):
    self.batch_dest_dir_split_is_custom = bool(
        (getattr(self, "batch_dest_dir_split", "") or "").strip()
    )


def _mark_split_name_r_custom(self, _context):
    self.split_name_r_is_custom = bool((self.split_name_r or "").strip())


def _mark_split_name_g_custom(self, _context):
    self.split_name_g_is_custom = bool((self.split_name_g or "").strip())


def _mark_split_name_b_custom(self, _context):
    self.split_name_b_is_custom = bool((self.split_name_b or "").strip())


def _mark_split_name_a_custom(self, _context):
    self.split_name_a_is_custom = bool((self.split_name_a or "").strip())


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def _unique_path_preserve_trailing_token(path: Path, tokens: tuple[str, ...]) -> Path:
    if not path.exists():
        return path
    stem = path.stem or ""
    stem_lower = stem.lower()
    token_match = None
    for token in sorted(tokens, key=len, reverse=True):
        if stem_lower.endswith(token.lower()):
            token_match = stem[-len(token) :]
            base = stem[: -len(token)]
            break
    if not token_match:
        return _unique_path(path)
    base = (base or "").rstrip("_")
    for i in range(1, 1000):
        if base:
            candidate_stem = f"{base}_{i}{token_match}"
        else:
            candidate_stem = f"{i}{token_match}"
        candidate = path.with_name(f"{candidate_stem}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


def _unique_image_name(blend_data, base_name: str) -> str:
    if base_name not in blend_data.images:
        return base_name
    for i in range(1, 1000):
        candidate = f"{base_name}.{i:03d}"
        if candidate not in blend_data.images:
            return candidate
    return base_name


def _image_abspath(img) -> str | None:
    if img is None:
        return None
    filepath = (
        getattr(img, "filepath_raw", "") or getattr(img, "filepath", "") or ""
    ).strip()
    if not filepath:
        return None
    try:
        return bpy.path.abspath(filepath)
    except Exception:
        return filepath


def _find_image_by_abspath(blend_data, abs_path: str):
    if not abs_path:
        return None
    for img in blend_data.images:
        try:
            existing = _image_abspath(img)
        except Exception:
            existing = None
        if existing and existing == abs_path:
            return img
    return None


def _ensure_pixels_loaded(img) -> bool:
    if img is None:
        return False
    try:
        if getattr(img, "source", "") == "FILE" and not getattr(img, "has_data", True):
            img.reload()
    except Exception:
        pass
    try:
        _ = img.pixels[0]
    except Exception:
        return False
    return True


# -----------------------------------------------------------------------------
# Utility: Update Callback to update Image Editor view when a channel image is changed.
# -----------------------------------------------------------------------------
def image_update_callback(prop_name):
    """
    Returns an update callback function for a pointer property to an image.
    When the property is updated, it searches for an IMAGE_EDITOR area and sets
    its active image to the one selected.
    """

    def update(self, context):
        img = getattr(self, prop_name)
        if img is not None:
            _set_active_image_in_image_editor(context, img)
            try:
                fp = (
                    getattr(img, "filepath_raw", "")
                    or getattr(img, "filepath", "")
                    or ""
                ).strip()
                src_path = None
                if fp:
                    fp = bpy.path.abspath(fp)
                    src_path = Path(fp)
                    if src_path.suffix:
                        packed_path = src_path.with_name(
                            f"{src_path.stem}_packed{src_path.suffix}"
                        )
                    else:
                        packed_path = src_path.with_name(f"{src_path.name}_packed.png")

                if prop_name == "rgb_image":
                    if src_path is not None and not getattr(
                        self, "save_path_single_is_custom", False
                    ):
                        self.save_path_single = str(packed_path)
                        self.save_path_single_is_custom = False
                if prop_name == "r_image":
                    if src_path is not None and not getattr(
                        self, "save_path_multi_is_custom", False
                    ):
                        self.save_path_multi = str(packed_path)
                        self.save_path_multi_is_custom = False
                if prop_name == "split_image":
                    if not getattr(self, "save_path_split_is_custom", False):
                        if src_path is not None:
                            if src_path.suffix:
                                split_path = src_path.with_name(
                                    f"{src_path.stem}_split{src_path.suffix}"
                                )
                            else:
                                split_path = src_path.with_name(
                                    f"{src_path.name}_split.png"
                                )
                            self.save_path_split = str(split_path)
                            self.save_path_split_is_custom = False
                    defaults = _default_split_output_names(img) or {}
                    if not getattr(self, "split_name_r_is_custom", False):
                        self.split_name_r = defaults.get("r", self.split_name_r)
                        self.split_name_r_is_custom = False
                    if not getattr(self, "split_name_g_is_custom", False):
                        self.split_name_g = defaults.get("g", self.split_name_g)
                        self.split_name_g_is_custom = False
                    if not getattr(self, "split_name_b_is_custom", False):
                        self.split_name_b = defaults.get("b", self.split_name_b)
                        self.split_name_b_is_custom = False
                    if not getattr(self, "split_name_a_is_custom", False):
                        self.split_name_a = defaults.get("a", self.split_name_a)
                        self.split_name_a_is_custom = False
            except Exception:
                pass

    return update


def _default_save_path_for_image(image, *, suffix: str) -> str | None:
    if image is None:
        return None
    filepath = (
        getattr(image, "filepath_raw", "") or getattr(image, "filepath", "") or ""
    ).strip()
    if not filepath:
        return None
    src_path = Path(bpy.path.abspath(filepath))
    if not src_path.name:
        return None
    if src_path.suffix:
        return str(src_path.with_name(f"{src_path.stem}_{suffix}{src_path.suffix}"))
    return str(src_path.with_name(f"{src_path.name}_{suffix}.png"))


def _default_split_output_names(image) -> dict[str, str] | None:
    if image is None:
        return None
    filepath = (
        getattr(image, "filepath_raw", "") or getattr(image, "filepath", "") or ""
    ).strip()
    if filepath:
        src_path = Path(bpy.path.abspath(filepath))
        base = src_path.stem or image.name
    else:
        base = image.name
    base = (base or "Image").strip()
    return {
        "r": f"{base}_red",
        "g": f"{base}_green",
        "b": f"{base}_blue",
        "a": f"{base}_alpha",
    }


def _ensure_default_save_paths(props) -> None:
    if not (getattr(props, "save_path_single", "") or "").strip() and props.rgb_image:
        suggested = _default_save_path_for_image(props.rgb_image, suffix="packed")
        if suggested:
            props.save_path_single = suggested
            props.save_path_single_is_custom = False
    if not (getattr(props, "save_path_multi", "") or "").strip() and props.r_image:
        suggested = _default_save_path_for_image(props.r_image, suffix="packed")
        if suggested:
            props.save_path_multi = suggested
            props.save_path_multi_is_custom = False
    if not (getattr(props, "save_path_split", "") or "").strip() and props.split_image:
        suggested = _default_save_path_for_image(props.split_image, suffix="split")
        if suggested:
            props.save_path_split = suggested
            props.save_path_split_is_custom = False
    if props.split_image:
        defaults = _default_split_output_names(props.split_image) or {}
        if not getattr(props, "split_name_r_is_custom", False):
            props.split_name_r = defaults.get("r", props.split_name_r)
            props.split_name_r_is_custom = False
        if not getattr(props, "split_name_g_is_custom", False):
            props.split_name_g = defaults.get("g", props.split_name_g)
            props.split_name_g_is_custom = False
        if not getattr(props, "split_name_b_is_custom", False):
            props.split_name_b = defaults.get("b", props.split_name_b)
            props.split_name_b_is_custom = False
        if not getattr(props, "split_name_a_is_custom", False):
            props.split_name_a = defaults.get("a", props.split_name_a)
            props.split_name_a_is_custom = False


def _sanitize_stem(value: str) -> str:
    stem = (value or "").strip()
    if not stem:
        return "Image"
    stem = re.sub(r"[\\s\\/\\\\:;*?\"<>|]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "Image"


def _batch_default_dir_for_image(img, folder_name: str) -> Path:
    abs_path = _image_abspath(img)
    if abs_path:
        try:
            src_dir = Path(abs_path).parent
        except Exception:
            src_dir = None
        else:
            if str(src_dir):
                return src_dir / folder_name
    blend_dir = bpy.path.abspath("//")
    if blend_dir:
        return Path(blend_dir) / folder_name
    return Path(bpy.app.tempdir or ".") / folder_name


def _batch_ext_for_image(img) -> str:
    abs_path = _image_abspath(img)
    if not abs_path:
        return ".png"
    ext = Path(abs_path).suffix.lower()
    return ext if _image_file_format_from_path(f"dummy{ext}") else ".png"


def _batch_base_stem_for_image(img) -> str:
    abs_path = _image_abspath(img)
    if abs_path:
        stem = Path(abs_path).stem
        if stem:
            return _sanitize_stem(stem)
    try:
        name = getattr(img, "name", "") or ""
    except Exception:
        name = ""
    return _sanitize_stem(name)


# -----------------------------------------------------------------------------
# Property Group: Stores all settings for channel packing.
# -----------------------------------------------------------------------------


class BeyondChannelPackerBatchImageItem(bpy.types.PropertyGroup):
    image: bpy.props.PointerProperty(
        name="Image",
        type=bpy.types.Image,
        description="Image datablock to use in batch processing",
    )
    filepath: bpy.props.StringProperty(
        name="Filepath",
        subtype="FILE_PATH",
        default="",
        options={"HIDDEN"},
    )


class BeyondChannelPackerProperties(bpy.types.PropertyGroup):
    # Mode selection: 'SINGLE' = one RGB image plus one (optional) alpha image.
    # 'MULTI' = four separate channel images.
    mode: bpy.props.EnumProperty(
        name="Mode",
        description="Select channel packing mode",
        items=[
            ("SINGLE", "RGB + Alpha", "Pack one RGB image and one Alpha image"),
            (
                "MULTI",
                "4-Channel (R+G+B+A)",
                "Pack four separate channel images (R, G, B, A)",
            ),
            (
                "SPLIT",
                "Split Channels",
                "Split a 3 or 4 channel image into separate channel images",
            ),
            (
                "BATCH",
                "Batch",
                "Process multiple images with shared settings",
            ),
        ],
        default="SINGLE",
    )

    # --- For SINGLE mode (RGB + Alpha) ---
    rgb_image: bpy.props.PointerProperty(
        name="RGB Image",
        type=bpy.types.Image,
        description="Image providing the RGB channels",
        update=image_update_callback("rgb_image"),
    )
    alpha_image: bpy.props.PointerProperty(
        name="Alpha Image",
        type=bpy.types.Image,
        description=(
            "Image providing the Alpha channel (will be flattened to one channel)"
        ),
        update=image_update_callback("alpha_image"),
    )
    override_img_if_exists: bpy.props.BoolProperty(
        name="Override Img if Exists",
        description="Overwrite the Save To file if it already exists",
        default=False,
    )

    # --- For MULTI mode (Separate Channels) ---
    r_image: bpy.props.PointerProperty(
        name="Red Channel Image",
        type=bpy.types.Image,
        description="Image providing the Red channel",
        update=image_update_callback("r_image"),
    )
    g_image: bpy.props.PointerProperty(
        name="Green Channel Image",
        type=bpy.types.Image,
        description="Image providing the Green channel",
        update=image_update_callback("g_image"),
    )
    b_image: bpy.props.PointerProperty(
        name="Blue Channel Image",
        type=bpy.types.Image,
        description="Image providing the Blue channel",
        update=image_update_callback("b_image"),
    )
    a_image: bpy.props.PointerProperty(
        name="Alpha Channel Image",
        type=bpy.types.Image,
        description="Image providing the Alpha channel",
        update=image_update_callback("a_image"),
    )

    split_image: bpy.props.PointerProperty(
        name="Split Source Image",
        type=bpy.types.Image,
        description="3 or 4 channel image to split into separate channel images",
        update=image_update_callback("split_image"),
    )
    split_name_r: bpy.props.StringProperty(
        name="Red",
        description="Output image name for Red channel",
        default="",
        update=_mark_split_name_r_custom,
    )
    split_name_r_is_custom: bpy.props.BoolProperty(
        name="Split Name R Is Custom",
        description="Internal flag for split name (r)",
        default=False,
        options={"HIDDEN"},
    )
    split_name_g: bpy.props.StringProperty(
        name="Green",
        description="Output image name for Green channel",
        default="",
        update=_mark_split_name_g_custom,
    )
    split_name_g_is_custom: bpy.props.BoolProperty(
        name="Split Name G Is Custom",
        description="Internal flag for split name (g)",
        default=False,
        options={"HIDDEN"},
    )
    split_name_b: bpy.props.StringProperty(
        name="Blue",
        description="Output image name for Blue channel",
        default="",
        update=_mark_split_name_b_custom,
    )
    split_name_b_is_custom: bpy.props.BoolProperty(
        name="Split Name B Is Custom",
        description="Internal flag for split name (b)",
        default=False,
        options={"HIDDEN"},
    )
    split_name_a: bpy.props.StringProperty(
        name="Alpha",
        description="Output image name for Alpha channel",
        default="",
        update=_mark_split_name_a_custom,
    )
    split_name_a_is_custom: bpy.props.BoolProperty(
        name="Split Name A Is Custom",
        description="Internal flag for split name (a)",
        default=False,
        options={"HIDDEN"},
    )

    # Storage for the resulting packed image.
    result_image: bpy.props.PointerProperty(
        name="Result Image",
        type=bpy.types.Image,
        description="The resulting packed image",
    )

    last_saved_path: bpy.props.StringProperty(
        name="Saved To",
        description="Last auto-saved packed image path",
        subtype="FILE_PATH",
        default="",
    )

    save_path_single: bpy.props.StringProperty(
        name="Save To",
        description="Base save target path for RGB + Alpha",
        subtype="FILE_PATH",
        default="",
        update=_mark_save_single_custom,
    )
    save_path_single_is_custom: bpy.props.BoolProperty(
        name="Save Path Single Is Custom",
        description="Internal flag for Save To (single)",
        default=False,
        options={"HIDDEN"},
    )

    save_path_multi: bpy.props.StringProperty(
        name="Save To",
        description="Base save target path for Separate Channels",
        subtype="FILE_PATH",
        default="",
        update=_mark_save_multi_custom,
    )
    save_path_multi_is_custom: bpy.props.BoolProperty(
        name="Save Path Multi Is Custom",
        description="Internal flag for Save To (multi)",
        default=False,
        options={"HIDDEN"},
    )

    save_path_split: bpy.props.StringProperty(
        name="Save To",
        description="Base save target path for Split Channels",
        subtype="FILE_PATH",
        default="",
        update=_mark_save_split_custom,
    )
    save_path_split_is_custom: bpy.props.BoolProperty(
        name="Save Path Split Is Custom",
        description="Internal flag for Save To (split)",
        default=False,
        options={"HIDDEN"},
    )

    auto_scale_to_largest: bpy.props.BoolProperty(
        name="Auto Scale to Largest",
        description=(
            "If images differ in size but have similar aspect ratios, "
            "scale them to the largest image for packing"
        ),
        default=True,
    )
    aspect_ratio_tolerance: bpy.props.FloatProperty(
        name="Aspect Tolerance",
        description=(
            "Allowed relative aspect ratio difference for auto scaling "
            "(e.g. 0.20 = 20%)"
        ),
        default=0.20,
        min=0.0,
        soft_max=0.1,
        max=0.5,
    )

    show_advanced: bpy.props.BoolProperty(
        name="Advanced Options",
        description="Show advanced output adjustments",
        default=False,
    )

    multi_gamma: bpy.props.FloatProperty(
        name="Gamma",
        description="Separate Channels output gamma adjustment",
        default=_MULTI_SRGB_GAMMA,
        min=0.1,
        soft_min=0.5,
        soft_max=2.0,
        max=4.0,
    )

    multi_contrast: bpy.props.FloatProperty(
        name="Contrast",
        description="Separate Channels output contrast (1.0 = no change)",
        default=_MULTI_SRGB_CONTRAST,
        min=0.0,
        soft_min=0.9,
        soft_max=1.1,
        max=2.0,
    )

    multi_brightness: bpy.props.FloatProperty(
        name="Brightness",
        description="Separate Channels brightness offset",
        default=_MULTI_SRGB_BRIGHTNESS,
        min=-1.0,
        soft_min=-0.2,
        soft_max=0.2,
        max=1.0,
    )

    # --- Batch mode ---
    batch_dest_dir_single: bpy.props.StringProperty(
        name="Destination Folder",
        description="Destination folder for RGB + Alpha batch output",
        subtype="DIR_PATH",
        default="",
        update=_mark_batch_dest_single_custom,
    )
    batch_dest_dir_single_is_custom: bpy.props.BoolProperty(
        name="Batch Dest Single Is Custom",
        description="Internal flag for batch destination (single)",
        default=False,
        options={"HIDDEN"},
    )
    batch_dest_dir_multi: bpy.props.StringProperty(
        name="Destination Folder",
        description="Destination folder for 4-channel batch output",
        subtype="DIR_PATH",
        default="",
        update=_mark_batch_dest_multi_custom,
    )
    batch_dest_dir_multi_is_custom: bpy.props.BoolProperty(
        name="Batch Dest Multi Is Custom",
        description="Internal flag for batch destination (multi)",
        default=False,
        options={"HIDDEN"},
    )
    batch_dest_dir_split: bpy.props.StringProperty(
        name="Destination Folder",
        description="Destination folder for split batch output",
        subtype="DIR_PATH",
        default="",
        update=_mark_batch_dest_split_custom,
    )
    batch_dest_dir_split_is_custom: bpy.props.BoolProperty(
        name="Batch Dest Split Is Custom",
        description="Internal flag for batch destination (split)",
        default=False,
        options={"HIDDEN"},
    )

    batch_show_single: bpy.props.BoolProperty(
        name="Batch: RGB + Alpha",
        description="Show/hide the RGB + Alpha batch section",
        default=True,
    )
    batch_show_multi: bpy.props.BoolProperty(
        name="Batch: 4-Channel (R+G+B+A)",
        description="Show/hide the 4-channel batch section",
        default=True,
    )
    batch_show_split: bpy.props.BoolProperty(
        name="Batch: Split Channels",
        description="Show/hide the split batch section",
        default=True,
    )

    batch_single_suffix_rgb: bpy.props.StringProperty(
        name="RGB Suffix",
        description="Suffix for the RGB images (e.g. _rgb). Can be empty",
        default="_rgb",
    )
    batch_single_suffix_alpha: bpy.props.StringProperty(
        name="Alpha Suffix",
        description="Suffix for the Alpha images (e.g. _a or _alpha)",
        default="_alpha",
    )

    batch_multi_suffix_r: bpy.props.StringProperty(
        name="R Suffix",
        description="Suffix for the R images (e.g. _r or _red)",
        default="_red",
    )
    batch_multi_suffix_g: bpy.props.StringProperty(
        name="G Suffix",
        description="Suffix for the G images (e.g. _g or _green)",
        default="_green",
    )
    batch_multi_suffix_b: bpy.props.StringProperty(
        name="B Suffix",
        description="Suffix for the B images (e.g. _b or _blue)",
        default="_blue",
    )
    batch_multi_suffix_a: bpy.props.StringProperty(
        name="A Suffix",
        description="Suffix for the A images (e.g. _a or _alpha)",
        default="_alpha",
    )

    batch_single_rgb_items: bpy.props.CollectionProperty(
        name="Batch RGB Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_single_rgb_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update(
            "batch_single_rgb_items", "batch_single_rgb_index"
        ),
    )
    batch_single_alpha_items: bpy.props.CollectionProperty(
        name="Batch Alpha Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_single_alpha_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update(
            "batch_single_alpha_items", "batch_single_alpha_index"
        ),
    )

    batch_multi_r_items: bpy.props.CollectionProperty(
        name="Batch R Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_multi_r_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update("batch_multi_r_items", "batch_multi_r_index"),
    )
    batch_multi_g_items: bpy.props.CollectionProperty(
        name="Batch G Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_multi_g_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update("batch_multi_g_items", "batch_multi_g_index"),
    )
    batch_multi_b_items: bpy.props.CollectionProperty(
        name="Batch B Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_multi_b_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update("batch_multi_b_items", "batch_multi_b_index"),
    )
    batch_multi_a_items: bpy.props.CollectionProperty(
        name="Batch A Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_multi_a_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update("batch_multi_a_items", "batch_multi_a_index"),
    )

    batch_split_items: bpy.props.CollectionProperty(
        name="Batch Split Images",
        type=BeyondChannelPackerBatchImageItem,
    )
    batch_split_index: bpy.props.IntProperty(
        default=0,
        update=_batch_list_index_update("batch_split_items", "batch_split_index"),
    )

    batch_keep_output_images_single: bpy.props.BoolProperty(
        name="Keep output images in Blender",
        description="Keep RGB + Alpha batch output images in Blender after processing",
        default=False,
    )
    batch_keep_output_images_multi: bpy.props.BoolProperty(
        name="Keep output images in Blender",
        description="Keep 4-channel batch output images in Blender after processing",
        default=False,
    )
    batch_keep_output_images_split: bpy.props.BoolProperty(
        name="Keep output images in Blender",
        description="Keep split batch output images in Blender after processing",
        default=False,
    )


# -----------------------------------------------------------------------------
# Operator: Pack Channels
# -----------------------------------------------------------------------------
class ChannelPackerOTPackChannels(bpy.types.Operator):
    """Pack selected channels into a new image"""

    bl_idname = "channelpacker.pack_channels"
    bl_label = "Pack Channels"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        mode = props.mode
        if mode == "BATCH":
            self.report({"ERROR"}, "Use the Batch tab for batch processing.")
            return {"CANCELLED"}
        props.last_saved_path = ""
        _ensure_default_save_paths(props)

        # -----------------------------------------------------------------------------
        # Determine target width/height.
        # If images differ in dimensions but have a similar aspect ratio, optionally
        # scale them to match the largest image for packing.
        # -----------------------------------------------------------------------------
        width = height = None
        target_img = None

        def get_selected_images():
            if mode == "SINGLE":
                return [img for img in [props.rgb_image, props.alpha_image] if img]
            return [
                img
                for img in [props.r_image, props.g_image, props.b_image, props.a_image]
                if img
            ]

        selected_images = get_selected_images()
        for img in selected_images:
            try:
                if getattr(img, "source", "") == "FILE" and not getattr(
                    img, "has_data", True
                ):
                    img.reload()
            except Exception:
                pass
            try:
                _ = img.pixels[0]
            except Exception:
                pass

        missing_sources: list[str] = []
        for img in selected_images:
            if img is None:
                continue
            abs_path = _image_abspath(img)
            if not abs_path:
                continue
            try:
                if getattr(img, "source", "") != "FILE":
                    continue
            except Exception:
                continue
            try:
                exists = Path(abs_path).exists()
            except Exception:
                exists = True
            if exists:
                continue
            try:
                packed = bool(getattr(img, "packed_file", None))
            except Exception:
                packed = False
            if packed:
                continue
            missing_sources.append(abs_path)

        if missing_sources:
            self.report(
                {"ERROR"},
                "Source image(s) missing on disk. Reload or re-link and try again.",
            )
            return {"CANCELLED"}

        # For SINGLE mode, the RGB image is required.
        if mode == "SINGLE":
            if props.rgb_image is None:
                self.report({"ERROR"}, "RGB Image is required for RGB + Alpha mode.")
                return {"CANCELLED"}
        else:
            if not selected_images:
                self.report(
                    {"ERROR"},
                    "At least one channel image must be provided in "
                    "Separate Channels mode.",
                )
                return {"CANCELLED"}

        if not selected_images:
            selected_images = get_selected_images()
        if not selected_images:
            self.report({"ERROR"}, "Could not determine image dimensions.")
            return {"CANCELLED"}

        if mode == "SINGLE" and props.rgb_image is not None:
            width, height = int(props.rgb_image.size[0]), int(props.rgb_image.size[1])
            target_img = props.rgb_image
        else:
            target_img = max(selected_images, key=lambda img: img.size[0] * img.size[1])
            width, height = int(target_img.size[0]), int(target_img.size[1])

        if width <= 0 or height <= 0:
            self.report({"ERROR"}, "Could not determine image dimensions.")
            return {"CANCELLED"}

        target_ratio = width / height
        tol = float(props.aspect_ratio_tolerance)
        for img in selected_images:
            iw, ih = int(img.size[0]), int(img.size[1])
            if iw == width and ih == height:
                continue
            if not props.auto_scale_to_largest:
                self.report({"ERROR"}, "All images must have the same dimensions.")
                return {"CANCELLED"}
            ratio = iw / ih if ih else 0.0
            if target_ratio:
                rel_diff = abs(ratio - target_ratio) / target_ratio
            else:
                rel_diff = 0.0
            if rel_diff > tol:
                self.report({"ERROR"}, "All images must have the same dimensions.")
                return {"CANCELLED"}

        if width is None or height is None:
            self.report({"ERROR"}, "Could not determine image dimensions.")
            return {"CANCELLED"}

        # -------------------------------------------------------------------------
        # Pack pixels using the older, known-good numpy approach.
        # -------------------------------------------------------------------------
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            self.report({"ERROR"}, "NumPy is required for packing in this version.")
            return {"CANCELLED"}

        output = np.zeros((height, width, 4), dtype=np.float32)
        output[:, :, 3] = 1.0

        def _resample_nearest(arr, src_w: int, src_h: int):
            if src_w == width and src_h == height:
                return arr
            y_idx = (np.arange(height) * (src_h / height)).astype(np.int32)
            x_idx = (np.arange(width) * (src_w / width)).astype(np.int32)
            return arr[y_idx[:, None], x_idx[None, :], :]

        def read_pixels_1d(img, *, treat_as_data: bool):
            _ensure_pixels_loaded(img)

            orig_colorspace = None
            orig_alpha_mode = None
            if treat_as_data:
                try:
                    orig_colorspace = img.colorspace_settings.name
                except Exception:
                    orig_colorspace = None
                try:
                    orig_alpha_mode = img.alpha_mode
                except Exception:
                    orig_alpha_mode = None
                try:
                    img.colorspace_settings.name = "Non-Color"
                except Exception:
                    pass
                try:
                    img.alpha_mode = "STRAIGHT"
                except Exception:
                    pass

            try:
                px_len = len(img.pixels)
            except Exception:
                px_len = 0
            if px_len <= 0:
                if treat_as_data:
                    try:
                        if orig_colorspace is not None:
                            img.colorspace_settings.name = orig_colorspace
                    except Exception:
                        pass
                    try:
                        if orig_alpha_mode is not None:
                            img.alpha_mode = orig_alpha_mode
                    except Exception:
                        pass
                return None

            arr = np.empty(px_len, dtype=np.float32)
            try:
                img.pixels.foreach_get(arr)
                return arr
            except Exception:
                try:
                    return np.array(img.pixels[:], dtype=np.float32)
                except Exception:
                    return None
            finally:
                if treat_as_data:
                    try:
                        if orig_colorspace is not None:
                            img.colorspace_settings.name = orig_colorspace
                    except Exception:
                        pass
                    try:
                        if orig_alpha_mode is not None:
                            img.alpha_mode = orig_alpha_mode
                    except Exception:
                        pass

        def get_pixels_array(img, *, treat_as_data: bool):
            if img is None:
                return None

            src_w, src_h = int(img.size[0]), int(img.size[1])
            if src_w <= 0 or src_h <= 0:
                self.report({"ERROR"}, f"Image '{img.name}' has invalid dimensions.")
                return None

            px = read_pixels_1d(img, treat_as_data=treat_as_data)
            if px is None:
                self.report({"ERROR"}, f"Image '{img.name}' has no pixel data.")
                return None

            denom = src_w * src_h
            if denom <= 0 or (px.size % denom) != 0:
                self.report(
                    {"ERROR"},
                    (
                        f"Image '{img.name}' pixel buffer size doesn't match its "
                        "dimensions."
                    ),
                )
                return None

            channels = int(px.size // denom)
            if channels <= 0:
                self.report({"ERROR"}, f"Image '{img.name}' has invalid pixel data.")
                return None

            try:
                arr = px.reshape((src_h, src_w, channels))
            except ValueError as exc:
                self.report({"ERROR"}, f"Error processing image '{img.name}': {exc}")
                return None

            if channels < 4:
                padded = np.zeros((src_h, src_w, 4), dtype=np.float32)
                padded[:, :, 3] = 1.0
                padded[:, :, 0:channels] = arr[:, :, 0:channels]
                if channels == 1:
                    padded[:, :, 1] = padded[:, :, 0]
                    padded[:, :, 2] = padded[:, :, 0]
                arr = padded
            else:
                arr = arr[:, :, 0:4]

            if src_w != width or src_h != height:
                arr = _resample_nearest(arr, src_w, src_h)
            return arr

        def extract_greyscale(arr, *, prefer_alpha: bool):
            if arr is None:
                return None

            rgb = arr[:, :, 0]
            if not prefer_alpha:
                return rgb

            alpha = arr[:, :, 3] if arr.shape[2] >= 4 else None
            if alpha is None:
                return rgb

            alpha_var = float(alpha.max() - alpha.min())
            rgb_var = float(rgb.max() - rgb.min())
            if alpha_var > 1e-6 and rgb_var <= 1e-6:
                return alpha
            return rgb

        if mode == "SINGLE":
            arr_rgb = get_pixels_array(props.rgb_image, treat_as_data=False)
            if arr_rgb is None:
                return {"CANCELLED"}
            output[:, :, 0] = arr_rgb[:, :, 0]
            output[:, :, 1] = arr_rgb[:, :, 1]
            output[:, :, 2] = arr_rgb[:, :, 2]

            if props.alpha_image is not None:
                arr_a = get_pixels_array(props.alpha_image, treat_as_data=False)
                alpha_data = extract_greyscale(arr_a, prefer_alpha=True)
                if alpha_data is None:
                    return {"CANCELLED"}
                output[:, :, 3] = alpha_data
            else:
                output[:, :, 3] = 1.0
        else:

            def read_channel(img):
                if img is None:
                    return None
                arr = get_pixels_array(img, treat_as_data=False)
                if arr is None:
                    return None
                data = extract_greyscale(arr, prefer_alpha=True)
                return data

            r_data = read_channel(props.r_image)
            g_data = read_channel(props.g_image)
            b_data = read_channel(props.b_image)
            a_data = read_channel(props.a_image)

            output[:, :, 0] = r_data if r_data is not None else 0.0
            output[:, :, 1] = g_data if g_data is not None else 0.0
            output[:, :, 2] = b_data if b_data is not None else 0.0
            output[:, :, 3] = a_data if a_data is not None else 1.0

            multi_gamma = float(getattr(props, "multi_gamma", _MULTI_SRGB_GAMMA))
            if multi_gamma <= 0.0:
                multi_gamma = 1.0
            multi_contrast = float(
                getattr(props, "multi_contrast", _MULTI_SRGB_CONTRAST)
            )
            multi_brightness = float(
                getattr(props, "multi_brightness", _MULTI_SRGB_BRIGHTNESS)
            )

            rgb = np.clip(output[:, :, 0:3], 0.0, 1.0)
            if abs(multi_gamma - 1.0) > 1e-6:
                rgb = np.power(rgb, multi_gamma)
            if abs(multi_contrast - 1.0) > 1e-9:
                rgb = (rgb - 0.5) * multi_contrast + 0.5
            if abs(multi_brightness) > 1e-9:
                rgb = rgb + multi_brightness
            output[:, :, 0:3] = np.clip(rgb, 0.0, 1.0)

        flat_pixels = output.astype(np.float32, copy=False).reshape(-1).tolist()

        def assign_pixels(img, pixels_list) -> str:
            from array import array

            expected = int(img.size[0]) * int(img.size[1]) * 4
            if expected <= 0:
                return "ERR:bad_size"
            if len(pixels_list) != expected:
                return f"ERR:len {len(pixels_list)} != {expected}"
            try:
                px_len = len(img.pixels)
            except Exception:
                px_len = -1
            if px_len not in {-1, expected}:
                return f"ERR:pixlen {px_len} != {expected}"

            try:
                img.pixels[:] = pixels_list
            except Exception:
                pass
            else:
                try:
                    img.update_tag()
                except Exception:
                    pass
                return "OK:slice"

            try:
                buf = array("f", pixels_list)
                img.pixels.foreach_set(buf)
            except Exception as exc:
                return f"ERR:set {type(exc).__name__}"
            try:
                img.update_tag()
            except Exception:
                pass
            return "OK:foreach"

        # -------------------------------------------------------------------------
        # Create/update the packed result image and save it.
        # -------------------------------------------------------------------------
        sources = (
            [props.rgb_image, props.alpha_image]
            if mode == "SINGLE"
            else [props.r_image, props.g_image, props.b_image, props.a_image]
        )
        override_existing = bool(props.override_img_if_exists)
        base_path = (
            props.save_path_single if mode == "SINGLE" else props.save_path_multi
        )
        out_path, file_format = _resolve_save_target_path(
            base_path=base_path,
            sources=sources,
            override_existing=override_existing,
        )

        base_target = (base_path or "").strip()
        if not base_target:
            base_target = _autosave_packed_target_for_sources(sources)
        base_target_path = Path(bpy.path.abspath(base_target))
        if not base_target_path.suffix:
            base_target_path = base_target_path.with_suffix(out_path.suffix)
        base_name = _result_image_base_name(mode=mode, base_stem=base_target_path.stem)

        ext = out_path.suffix.lower()
        wants_float = ext in {".exr", ".hdr"}
        for src in sources:
            if src is None:
                continue
            if bool(getattr(src, "is_float", False)):
                wants_float = True
                break

        base_img = context.blend_data.images.get(base_name)
        if base_img is None and mode == "MULTI":
            base_img = _find_image_by_abspath(context.blend_data, str(out_path))

        if base_img is None and mode == "MULTI":
            if out_path.exists():
                try:
                    base_img = context.blend_data.images.load(
                        str(out_path),
                        check_existing=True,
                    )
                except Exception:
                    base_img = None

        if base_img is None:
            result_img = context.blend_data.images.new(
                name=_unique_image_name(context.blend_data, base_name),
                width=width,
                height=height,
                alpha=True,
                float_buffer=wants_float,
            )
        else:
            result_img = base_img

        try:
            result_img.name = base_name
        except Exception:
            pass
        try:
            result_img.use_fake_user = False
        except Exception:
            pass
        if int(result_img.size[0]) != width or int(result_img.size[1]) != height:
            desired_path = str(out_path)
            original_colorspace = None
            original_alpha_mode = None
            try:
                original_colorspace = result_img.colorspace_settings.name
            except Exception:
                original_colorspace = None
            try:
                original_alpha_mode = result_img.alpha_mode
            except Exception:
                original_alpha_mode = None

            try:
                needs_fallback = bool(
                    getattr(result_img, "source", "") == "FILE"
                    and not getattr(result_img, "has_data", True)
                )
            except Exception:
                needs_fallback = False

            fallback = None
            if needs_fallback:
                for src in sources:
                    fp = _image_abspath(src)
                    if not fp:
                        continue
                    try:
                        if Path(fp).exists():
                            fallback = fp
                            break
                    except Exception:
                        fallback = fp
                        break
                if fallback:
                    try:
                        result_img.filepath_raw = fallback
                    except Exception:
                        fallback = None
                if fallback:
                    try:
                        result_img.reload()
                    except Exception:
                        pass

            _ensure_pixels_loaded(result_img)
            try:
                result_img.scale(width, height)
            except Exception as exc:
                self.report(
                    {"ERROR"},
                    f"Could not scale result image '{result_img.name}': {exc}",
                )
                return {"CANCELLED"}
            try:
                result_img.filepath_raw = desired_path
            except Exception:
                pass
            try:
                if original_colorspace is not None:
                    result_img.colorspace_settings.name = original_colorspace
            except Exception:
                pass
            try:
                if original_alpha_mode is not None:
                    result_img.alpha_mode = original_alpha_mode
            except Exception:
                pass

        try:
            result_img.use_view_as_render = False
        except Exception:
            pass
        try:
            result_img.alpha_mode = "STRAIGHT"
        except Exception:
            pass
        try:
            result_img.colorspace_settings.name = "sRGB"
        except Exception:
            pass

        set_status = assign_pixels(result_img, flat_pixels)
        if set_status.startswith("ERR:"):
            self.report({"ERROR"}, f"Could not write packed pixels: {set_status}")
            return {"CANCELLED"}
        props.result_image = result_img
        try:
            props.result_image.update()
        except Exception:
            pass

        props.result_image.filepath_raw = str(out_path)
        try:
            props.result_image.file_format = file_format
        except Exception:
            pass
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            props.result_image.save(filepath=str(out_path), save_copy=False)
            if not out_path.exists():
                raise RuntimeError("File was not written to disk.")
            try:
                props.result_image.filepath = str(out_path)
            except Exception:
                pass
            props.last_saved_path = str(out_path)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Packed, but could not save result: {exc}")
            return {"CANCELLED"}

        _set_active_image_in_image_editor(context, props.result_image)

        if not (props.last_saved_path or "").strip():
            self.report({"ERROR"}, "Packing finished, but did not save to disk.")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Channel packing complete. Saved: {props.last_saved_path}",
        )
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Operator: Split Channels
# -----------------------------------------------------------------------------
class ChannelPackerOTSplitChannels(bpy.types.Operator):
    """Split a 3 or 4 channel image into separate channel images"""

    bl_idname = "channelpacker.split_channels"
    bl_label = "Split Channels"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        props.last_saved_path = ""
        _ensure_default_save_paths(props)

        src_img = props.split_image
        if src_img is None:
            self.report({"ERROR"}, "Split Source Image is required.")
            return {"CANCELLED"}

        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            self.report({"ERROR"}, "NumPy is required for splitting in this version.")
            return {"CANCELLED"}

        try:
            if getattr(src_img, "source", "") == "FILE" and not getattr(
                src_img, "has_data", True
            ):
                src_img.reload()
        except Exception:
            pass
        try:
            _ = src_img.pixels[0]
        except Exception:
            pass

        width, height = int(src_img.size[0]), int(src_img.size[1])
        if width <= 0 or height <= 0:
            self.report({"ERROR"}, "Split Source Image has invalid dimensions.")
            return {"CANCELLED"}

        try:
            px_len = len(src_img.pixels)
        except Exception:
            px_len = 0
        if px_len <= 0:
            self.report({"ERROR"}, "Split Source Image has no pixel data.")
            return {"CANCELLED"}

        flat = np.empty(px_len, dtype=np.float32)
        try:
            src_img.pixels.foreach_get(flat)
        except Exception:
            try:
                flat = np.array(src_img.pixels[:], dtype=np.float32)
            except Exception:
                self.report({"ERROR"}, "Could not read Split Source Image pixels.")
                return {"CANCELLED"}

        denom = width * height
        if denom <= 0 or (flat.size % denom) != 0:
            self.report({"ERROR"}, "Split Source Image pixel buffer mismatch.")
            return {"CANCELLED"}

        channels = int(flat.size // denom)
        if channels < 3:
            self.report({"ERROR"}, "Split Source Image must have 3 or 4 channels.")
            return {"CANCELLED"}

        arr = flat.reshape((height, width, channels))
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        a = arr[:, :, 3] if channels >= 4 else None

        base = (props.save_path_split or "").strip()
        base_target = base
        if not base_target:
            base_target = _default_save_path_for_image(
                src_img, suffix="split"
            ) or _autosave_target_for_sources([src_img])
        base_path = Path(bpy.path.abspath(base_target))
        file_format, ext = _autosave_format_for_path(str(base_path))
        if base_path.suffix:
            detected = _image_file_format_from_path(str(base_path))
            if detected is not None:
                file_format = detected
                ext = base_path.suffix
        if not base_path.suffix:
            base_path = base_path.with_suffix(ext)

        wants_float = ext.lower() in {".exr", ".hdr"} or bool(
            getattr(src_img, "is_float", False)
        )
        override_existing = bool(props.override_img_if_exists)

        def _clean_stem(value: str, fallback: str) -> str:
            stem = (value or "").strip()
            if not stem:
                return fallback
            parsed = Path(stem)
            if parsed.suffix and _image_file_format_from_path(stem) is not None:
                return parsed.stem or fallback
            return stem

        base_dir = base_path.parent if str(base_path.parent) else Path(".")
        base_stem = base_path.stem or (src_img.name or "Image")
        defaults = _default_split_output_names(src_img) or {}
        name_r = _clean_stem(props.split_name_r, defaults.get("r", f"{base_stem}_red"))
        name_g = _clean_stem(
            props.split_name_g,
            defaults.get("g", f"{base_stem}_green"),
        )
        name_b = _clean_stem(props.split_name_b, defaults.get("b", f"{base_stem}_blue"))
        name_a = _clean_stem(
            props.split_name_a,
            defaults.get("a", f"{base_stem}_alpha"),
        )

        channel_defs = [(name_r, r), (name_g, g), (name_b, b)]
        if a is not None:
            channel_defs.append((name_a, a))

        saved_paths: list[Path] = []
        preview_img = None
        try:
            for stem, channel_data in channel_defs:
                out_path = base_dir / f"{stem}{ext}"
                if not override_existing:
                    out_path = _unique_path_preserve_trailing_token(
                        out_path,
                        (
                            "_alpha",
                            "_green",
                            "_blue",
                            "_red",
                            "_a",
                            "_g",
                            "_b",
                            "_r",
                        ),
                    )

                out_abs = str(out_path)
                out_img = context.blend_data.images.get(stem)
                if out_img is None:
                    out_img = _find_image_by_abspath(context.blend_data, out_abs)
                if out_img is None:
                    out_img = context.blend_data.images.new(
                        name=stem,
                        width=width,
                        height=height,
                        alpha=True,
                        float_buffer=wants_float,
                    )
                else:
                    _ensure_pixels_loaded(out_img)
                    try:
                        out_img.name = stem
                    except Exception:
                        pass
                    if int(out_img.size[0]) != width or int(out_img.size[1]) != height:
                        try:
                            out_img.scale(width, height)
                        except Exception:
                            pass
                if preview_img is None:
                    preview_img = out_img

                try:
                    out_img.use_view_as_render = False
                except Exception:
                    pass
                try:
                    out_img.alpha_mode = "STRAIGHT"
                except Exception:
                    pass
                try:
                    out_img.colorspace_settings.name = "Non-Color"
                except Exception:
                    pass

                vals = channel_data.astype(np.float32, copy=False).reshape(-1)
                out_px = np.empty(vals.size * 4, dtype=np.float32)
                out_px[0::4] = vals
                out_px[1::4] = vals
                out_px[2::4] = vals
                out_px[3::4] = 1.0

                try:
                    out_img.pixels.foreach_set(out_px)
                except Exception:
                    out_img.pixels[:] = out_px.tolist()
                try:
                    out_img.update()
                except Exception:
                    pass

                out_img.filepath_raw = str(out_path)
                try:
                    out_img.file_format = file_format
                except Exception:
                    pass
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_img.save(
                    filepath=str(out_path),
                    save_copy=not override_existing,
                )
                saved_paths.append(out_path)

            if preview_img is not None:
                props.result_image = preview_img
            if saved_paths:
                props.last_saved_path = str(saved_paths[-1])

        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not save split channels: {exc}")
            return {"CANCELLED"}

        if props.result_image is not None:
            _set_active_image_in_image_editor(context, props.result_image)
        self.report({"INFO"}, f"Split complete. Saved {len(saved_paths)} image(s).")
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Operator: Load Image From Disk (File Browser)
# -----------------------------------------------------------------------------
class ChannelPackerOTLoadImage(bpy.types.Operator):
    """Load an image from disk and assign it to a channel slot"""

    bl_idname = "channelpacker.load_image"
    bl_label = "Load Image"
    bl_options = {"REGISTER", "UNDO"}

    target_prop: bpy.props.StringProperty(name="Target Property")
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.bmp;*.exr;*.hdr",
        options={"HIDDEN"},
    )

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        allowed = {
            "rgb_image",
            "alpha_image",
            "r_image",
            "g_image",
            "b_image",
            "a_image",
            "split_image",
        }
        if self.target_prop not in allowed:
            self.report({"ERROR"}, "Invalid target image slot.")
            return {"CANCELLED"}

        props = context.scene.beyond_channel_packer
        try:
            img = context.blend_data.images.load(self.filepath, check_existing=True)
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not load image: {exc}")
            return {"CANCELLED"}

        setattr(props, self.target_prop, img)
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Batch Mode: UI Lists + Operators
# -----------------------------------------------------------------------------

_BATCH_LIST_PROPS: dict[str, str] = {
    "batch_single_rgb_items": "batch_single_rgb_index",
    "batch_single_alpha_items": "batch_single_alpha_index",
    "batch_multi_r_items": "batch_multi_r_index",
    "batch_multi_g_items": "batch_multi_g_index",
    "batch_multi_b_items": "batch_multi_b_index",
    "batch_multi_a_items": "batch_multi_a_index",
    "batch_split_items": "batch_split_index",
}


class ChannelPackerULBatchImages(bpy.types.UIList):
    bl_idname = "CHANNELPACKER_UL_batch_images"

    def draw_item(
        self,
        _context,
        layout,
        _data,
        item,
        _icon,
        _active_data,
        _active_propname,
        _index,
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.prop(item, "image", text="")
        else:
            layout.alignment = "CENTER"
            layout.label(text="", icon="IMAGE_DATA")


class ChannelPackerOTBatchListAdd(bpy.types.Operator):
    """Add one or more images to a batch list"""

    bl_idname = "channelpacker.batch_list_add"
    bl_label = "Add Images"
    bl_options = {"REGISTER", "UNDO"}

    collection_prop: bpy.props.StringProperty(name="Collection Property")
    fill_other_channels: bpy.props.BoolProperty(
        name="Fill lists for other channels based on suffix",
        description=(
            "If enabled, tries to find matching images in the same folder(s) and "
            "adds them to the other channel lists based on suffix settings"
        ),
        default=False,
    )
    name_contains: bpy.props.StringProperty(
        name="Text",
        description="Used by the select-matching operator to select many files at once",
        default="",
    )
    name_contains_case_sensitive: bpy.props.BoolProperty(
        name="Case Sensitive",
        description="Match filename contains with case sensitivity",
        default=False,
    )

    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.bmp;*.exr;*.hdr",
        options={"HIDDEN"},
    )
    autodetect_key: bpy.props.StringProperty(default="", options={"HIDDEN"})

    def invoke(self, context, _event):
        props = context.scene.beyond_channel_packer
        self._ensure_suffix_defaults(props)
        if not (self.name_contains or "").strip():
            self.name_contains = self._default_name_contains_for_list(props)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, _context):
        self.autodetect_key = ""

    def draw(self, context):
        layout = self.layout
        props = context.scene.beyond_channel_packer

        box = layout.box()
        box.label(text="Add all images in folder that contain")
        sub = box.column(align=True)
        sub.prop(self, "name_contains")
        sub.prop(self, "name_contains_case_sensitive")
        op = sub.operator(
            "channelpacker.filebrowser_select_matching",
            text="Select Matching Files",
            icon="RESTRICT_SELECT_OFF",
        )
        op.name_contains = self.name_contains
        op.case_sensitive = self.name_contains_case_sensitive
        op.extend_selection = False

        layout.prop(
            self,
            "fill_other_channels",
            text="Auto-fill other channel lists based on suffix",
        )

        col = layout.column(align=True)
        col.label(text="Suffix Settings (leave empty to auto-detect)")
        col.enabled = self.fill_other_channels
        if self.collection_prop in {
            "batch_single_rgb_items",
            "batch_single_alpha_items",
        }:
            col.prop(props, "batch_single_suffix_rgb")
            col.prop(props, "batch_single_suffix_alpha")
            return
        if self.collection_prop in {
            "batch_multi_r_items",
            "batch_multi_g_items",
            "batch_multi_b_items",
            "batch_multi_a_items",
        }:
            col.prop(props, "batch_multi_suffix_r")
            col.prop(props, "batch_multi_suffix_g")
            col.prop(props, "batch_multi_suffix_b")
            col.prop(props, "batch_multi_suffix_a")

    def _ensure_suffix_defaults(self, props) -> None:
        if self.collection_prop in {
            "batch_multi_r_items",
            "batch_multi_g_items",
            "batch_multi_b_items",
            "batch_multi_a_items",
        }:
            # Default to the verbose color words family.
            if not (props.batch_multi_suffix_r or "").strip():
                props.batch_multi_suffix_r = "_red"
            if not (props.batch_multi_suffix_g or "").strip():
                props.batch_multi_suffix_g = "_green"
            if not (props.batch_multi_suffix_b or "").strip():
                props.batch_multi_suffix_b = "_blue"
            if not (props.batch_multi_suffix_a or "").strip():
                props.batch_multi_suffix_a = "_alpha"
            return

        if self.collection_prop in {
            "batch_single_rgb_items",
            "batch_single_alpha_items",
        }:
            if not (props.batch_single_suffix_rgb or "").strip():
                props.batch_single_suffix_rgb = "_rgb"
            if not (props.batch_single_suffix_alpha or "").strip():
                props.batch_single_suffix_alpha = "_alpha"

    def _default_name_contains_for_list(self, props) -> str:
        prop = self.collection_prop
        if prop == "batch_multi_r_items":
            return (props.batch_multi_suffix_r or "_red").strip() or "_red"
        if prop == "batch_multi_g_items":
            return (props.batch_multi_suffix_g or "_green").strip() or "_green"
        if prop == "batch_multi_b_items":
            return (props.batch_multi_suffix_b or "_blue").strip() or "_blue"
        if prop == "batch_multi_a_items":
            return (props.batch_multi_suffix_a or "_alpha").strip() or "_alpha"
        if prop == "batch_single_rgb_items":
            return (props.batch_single_suffix_rgb or "_rgb").strip() or "_rgb"
        if prop == "batch_single_alpha_items":
            return (props.batch_single_suffix_alpha or "_alpha").strip() or "_alpha"
        return ""

    def check(self, context):
        props = context.scene.beyond_channel_packer
        directory = (self.directory or "").strip()
        first_name = ""
        if self.files:
            try:
                first_name = (self.files[0].name or "").strip()
            except Exception:
                first_name = ""
        key = f"{self.collection_prop}|{directory}|{first_name}"
        if not first_name or key == (getattr(self, "autodetect_key", "") or ""):
            return False
        self.autodetect_key = key

        stem = Path(first_name).stem
        changed = False

        if self.collection_prop in {
            "batch_multi_r_items",
            "batch_multi_g_items",
            "batch_multi_b_items",
            "batch_multi_a_items",
        }:
            detected = _detect_rgba_suffixes_any(stem)
            if detected is None:
                return False
            current = {
                "r": props.batch_multi_suffix_r,
                "g": props.batch_multi_suffix_g,
                "b": props.batch_multi_suffix_b,
                "a": props.batch_multi_suffix_a,
            }
            if _is_known_rgba_suffix_set(current) or any(
                not (v or "").strip() for v in current.values()
            ):
                if props.batch_multi_suffix_r != detected["r"]:
                    props.batch_multi_suffix_r = detected["r"]
                    changed = True
                if props.batch_multi_suffix_g != detected["g"]:
                    props.batch_multi_suffix_g = detected["g"]
                    changed = True
                if props.batch_multi_suffix_b != detected["b"]:
                    props.batch_multi_suffix_b = detected["b"]
                    changed = True
                if props.batch_multi_suffix_a != detected["a"]:
                    props.batch_multi_suffix_a = detected["a"]
                    changed = True
            return changed

        if self.collection_prop in {
            "batch_single_rgb_items",
            "batch_single_alpha_items",
        }:
            active_kind = (
                "rgb" if self.collection_prop == "batch_single_rgb_items" else "alpha"
            )
            detected_pair = _detect_single_suffixes(stem, active_kind)
            if detected_pair is None:
                return False
            rgb_suffix, alpha_suffix = detected_pair
            current_rgb = props.batch_single_suffix_rgb
            current_alpha = props.batch_single_suffix_alpha
            if _is_known_single_suffix_set(current_rgb, current_alpha) or (
                not (current_rgb or "").strip() or not (current_alpha or "").strip()
            ):
                if rgb_suffix and props.batch_single_suffix_rgb != rgb_suffix:
                    props.batch_single_suffix_rgb = rgb_suffix
                    changed = True
                if alpha_suffix and props.batch_single_suffix_alpha != alpha_suffix:
                    props.batch_single_suffix_alpha = alpha_suffix
                    changed = True
            return changed

        return False

    def execute(self, context):
        if self.collection_prop not in _BATCH_LIST_PROPS:
            self.report({"ERROR"}, "Invalid batch list.")
            return {"CANCELLED"}

        props = context.scene.beyond_channel_packer
        items = getattr(props, self.collection_prop)
        index_prop = _BATCH_LIST_PROPS[self.collection_prop]

        filepaths = [str(p) for p in self._gather_selected_paths()]

        existing_paths = {
            (getattr(it, "filepath", "") or "").lower()
            for it in items
            if it is not None
        }
        filepaths = [fp for fp in filepaths if str(fp).lower() not in existing_paths]

        added = 0
        first_added_image = None
        selected_paths: list[Path] = []
        for fp in filepaths:
            selected_paths.append(Path(fp))
            try:
                img = context.blend_data.images.load(fp, check_existing=True)
            except RuntimeError as exc:
                self.report({"WARNING"}, f"Could not load image: {exc}")
                continue
            item = items.add()
            item.image = img
            item.filepath = fp
            if first_added_image is None:
                first_added_image = img
            added += 1

        if added:
            setattr(props, index_prop, max(0, len(items) - 1))
            if first_added_image is not None:
                if self.collection_prop in {
                    "batch_single_rgb_items",
                    "batch_single_alpha_items",
                }:
                    _batch_maybe_set_dest_dir(
                        props,
                        dest_attr="batch_dest_dir_single",
                        folder_name="Packed_Batch",
                        image=first_added_image,
                    )
                elif self.collection_prop in {
                    "batch_multi_r_items",
                    "batch_multi_g_items",
                    "batch_multi_b_items",
                    "batch_multi_a_items",
                }:
                    _batch_maybe_set_dest_dir(
                        props,
                        dest_attr="batch_dest_dir_multi",
                        folder_name="Packed_Batch",
                        image=first_added_image,
                    )
                elif self.collection_prop == "batch_split_items":
                    _batch_maybe_set_dest_dir(
                        props,
                        dest_attr="batch_dest_dir_split",
                        folder_name="Split_Batch",
                        image=first_added_image,
                    )

            if self.fill_other_channels and selected_paths:
                self._fill_other_lists(context, props, selected_paths)
            return {"FINISHED"}

        self.report({"WARNING"}, "No images were added.")
        return {"CANCELLED"}

    def _gather_selected_paths(self) -> list[Path]:
        paths: list[Path] = []

        directory_raw = (self.directory or "").strip()
        base_dir = Path(directory_raw) if directory_raw else None

        def is_abs_name(name: str) -> bool:
            if not name:
                return False
            if (len(name) >= 2 and name[1] == ":") or name.startswith(("/", "\\")):
                return True
            return False

        def add_part(part: str, *, override_base: Path | None = None) -> None:
            part = (part or "").strip()
            if not part:
                return
            if is_abs_name(part):
                paths.append(Path(part))
                return
            use_base = override_base or base_dir
            if use_base is not None:
                paths.append(use_base / part)
            else:
                paths.append(Path(part))

        if self.files:
            for entry in self.files:
                name = (getattr(entry, "name", "") or "").strip()
                if not name:
                    continue
                # Some contexts end up with a single entry containing ';' joined names.
                if ";" in name:
                    for part in name.split(";"):
                        add_part(part)
                else:
                    add_part(name)
            return paths

        fp = (self.filepath or "").strip()
        if not fp:
            return paths
        if ";" in fp:
            parts = [p.strip() for p in fp.split(";") if p.strip()]
            if not parts:
                return paths
            first = parts[0]
            if is_abs_name(first):
                first_path = Path(first)
                paths.append(first_path)
                first_dir = first_path.parent
                for part in parts[1:]:
                    add_part(part, override_base=first_dir)
            else:
                # If we got only names, treat them as in `directory`.
                for part in parts:
                    add_part(part)
            return paths

        add_part(fp)
        return paths

    def _fill_other_lists(self, context, props, selected_paths: list[Path]) -> None:
        if self.collection_prop in {
            "batch_single_rgb_items",
            "batch_single_alpha_items",
        }:
            active_kind = (
                "rgb" if self.collection_prop == "batch_single_rgb_items" else "alpha"
            )
            rgb_suffix = (props.batch_single_suffix_rgb or "").strip()
            alpha_suffix = (props.batch_single_suffix_alpha or "").strip()
            if not rgb_suffix and not alpha_suffix and selected_paths:
                detected = _detect_single_suffixes(selected_paths[0].stem, active_kind)
                if detected is not None:
                    rgb_suffix, alpha_suffix = detected
            if not alpha_suffix:
                alpha_suffix = "_a"
            if active_kind == "alpha" and not alpha_suffix:
                alpha_suffix = "_a"
            if not (props.batch_single_suffix_rgb or "").strip():
                props.batch_single_suffix_rgb = rgb_suffix
            if not (props.batch_single_suffix_alpha or "").strip():
                props.batch_single_suffix_alpha = alpha_suffix

            other_prop = (
                "batch_single_alpha_items"
                if active_kind == "rgb"
                else "batch_single_rgb_items"
            )
            other_items = getattr(props, other_prop)
            for path in selected_paths:
                stem = path.stem
                if active_kind == "rgb":
                    base = _strip_suffix_case_insensitive(stem, rgb_suffix)
                    suffix = alpha_suffix
                else:
                    base = _strip_suffix_case_insensitive(stem, alpha_suffix)
                    suffix = rgb_suffix
                directory = path.parent
                files_map = _batch_dir_file_map(directory)
                candidate_names = [f"{base}{suffix}{path.suffix}"]
                if path.suffix.lower() not in _BATCH_ALLOWED_EXTS:
                    candidate_names = [f"{base}{suffix}.png"]
                chosen = None
                for name in candidate_names:
                    chosen = files_map.get(name.lower())
                    if chosen is not None:
                        break
                if chosen is None:
                    for ext in _BATCH_ALLOWED_EXTS:
                        name = f"{base}{suffix}{ext}"
                        chosen = files_map.get(name.lower())
                        if chosen is not None:
                            break
                item = other_items.add()
                if chosen is not None:
                    try:
                        img = context.blend_data.images.load(
                            str(chosen), check_existing=True
                        )
                    except RuntimeError:
                        img = None
                    item.image = img
                    item.filepath = str(chosen)
                else:
                    item.image = None
                    item.filepath = str(directory / f"{base}{suffix}{path.suffix}")
            return

        multi_map = {
            "batch_multi_r_items": "r",
            "batch_multi_g_items": "g",
            "batch_multi_b_items": "b",
            "batch_multi_a_items": "a",
        }
        if self.collection_prop not in multi_map:
            return
        active_channel = multi_map[self.collection_prop]
        suffixes = {
            "r": (props.batch_multi_suffix_r or "").strip(),
            "g": (props.batch_multi_suffix_g or "").strip(),
            "b": (props.batch_multi_suffix_b or "").strip(),
            "a": (props.batch_multi_suffix_a or "").strip(),
        }
        if not any(suffixes.values()) and selected_paths:
            detected = _detect_rgba_suffixes_for_channel(
                selected_paths[0].stem, active_channel
            )
            if detected is not None:
                suffixes.update(detected)
        if not suffixes["r"]:
            suffixes["r"] = "_r"
        if not suffixes["g"]:
            suffixes["g"] = _apply_suffix_case("_g", suffixes["r"])
        if not suffixes["b"]:
            suffixes["b"] = _apply_suffix_case("_b", suffixes["r"])
        if not suffixes["a"]:
            suffixes["a"] = _apply_suffix_case("_a", suffixes["r"])
        if not (props.batch_multi_suffix_r or "").strip():
            props.batch_multi_suffix_r = suffixes["r"]
        if not (props.batch_multi_suffix_g or "").strip():
            props.batch_multi_suffix_g = suffixes["g"]
        if not (props.batch_multi_suffix_b or "").strip():
            props.batch_multi_suffix_b = suffixes["b"]
        if not (props.batch_multi_suffix_a or "").strip():
            props.batch_multi_suffix_a = suffixes["a"]

        list_props = {
            "r": "batch_multi_r_items",
            "g": "batch_multi_g_items",
            "b": "batch_multi_b_items",
            "a": "batch_multi_a_items",
        }
        other_channels = [ch for ch in ("r", "g", "b", "a") if ch != active_channel]
        other_lists = {ch: getattr(props, list_props[ch]) for ch in other_channels}
        active_suffix = suffixes[active_channel]
        for path in selected_paths:
            stem = path.stem
            base = _strip_suffix_case_insensitive(stem, active_suffix)
            directory = path.parent
            files_map = _batch_dir_file_map(directory)
            for ch in other_channels:
                suffix = suffixes[ch]
                chosen = None
                name = f"{base}{suffix}{path.suffix}"
                chosen = files_map.get(name.lower())
                if chosen is None:
                    for ext in _BATCH_ALLOWED_EXTS:
                        name = f"{base}{suffix}{ext}"
                        chosen = files_map.get(name.lower())
                        if chosen is not None:
                            break
                item = other_lists[ch].add()
                if chosen is not None:
                    try:
                        img = context.blend_data.images.load(
                            str(chosen), check_existing=True
                        )
                    except RuntimeError:
                        img = None
                    item.image = img
                    item.filepath = str(chosen)
                else:
                    item.image = None
                    item.filepath = str(directory / f"{base}{suffix}{path.suffix}")


class ChannelPackerOTBatchListRemove(bpy.types.Operator):
    """Remove the active item from a batch list"""

    bl_idname = "channelpacker.batch_list_remove"
    bl_label = "Remove Image"
    bl_options = {"REGISTER", "UNDO"}

    collection_prop: bpy.props.StringProperty(name="Collection Property")

    def execute(self, context):
        if self.collection_prop not in _BATCH_LIST_PROPS:
            self.report({"ERROR"}, "Invalid batch list.")
            return {"CANCELLED"}

        props = context.scene.beyond_channel_packer
        items = getattr(props, self.collection_prop)
        index_prop = _BATCH_LIST_PROPS[self.collection_prop]
        idx = int(getattr(props, index_prop, 0) or 0)
        if idx < 0 or idx >= len(items):
            return {"CANCELLED"}

        items.remove(idx)
        setattr(props, index_prop, min(idx, max(0, len(items) - 1)))
        return {"FINISHED"}


class ChannelPackerOTBatchListClear(bpy.types.Operator):
    """Clear all items from a batch list"""

    bl_idname = "channelpacker.batch_list_clear"
    bl_label = "Clear Images"
    bl_options = {"REGISTER", "UNDO"}

    collection_prop: bpy.props.StringProperty(name="Collection Property")
    remove_from_blender: bpy.props.BoolProperty(
        name="Also remove cleared images from Blender?",
        description="If enabled, removes the cleared image datablocks from this .blend (files on disk are not deleted)",
        default=False,
    )

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "remove_from_blender")

    def execute(self, context):
        if self.collection_prop not in _BATCH_LIST_PROPS:
            self.report({"ERROR"}, "Invalid batch list.")
            return {"CANCELLED"}

        props = context.scene.beyond_channel_packer
        items = getattr(props, self.collection_prop)
        index_prop = _BATCH_LIST_PROPS[self.collection_prop]
        image_names: list[str] = []
        if self.remove_from_blender:
            seen: set[str] = set()
            for it in items:
                img = getattr(it, "image", None)
                if img is None:
                    continue
                name = (getattr(img, "name", "") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                image_names.append(name)
        items.clear()
        setattr(props, index_prop, 0)
        if self.remove_from_blender and image_names:
            for name in image_names:
                img = context.blend_data.images.get(name)
                _batch_remove_output_image(context.blend_data, img)
        return {"FINISHED"}


class ChannelPackerOTFileBrowserSelectMatching(bpy.types.Operator):
    """Select files in the current File Browser folder by substring match"""

    bl_idname = "channelpacker.filebrowser_select_matching"
    bl_label = "Select Matching Files"
    bl_options = {"REGISTER", "UNDO"}

    name_contains: bpy.props.StringProperty(name="Filename Contains", default="")
    case_sensitive: bpy.props.BoolProperty(name="Case Sensitive", default=False)
    extend_selection: bpy.props.BoolProperty(
        name="Extend Selection",
        description="Keep existing selection and add matching files",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        area = getattr(context, "area", None)
        if area is None:
            return False
        return getattr(area, "type", "") == "FILE_BROWSER"

    def execute(self, context):
        needle = (self.name_contains or "").strip()
        if not needle:
            self.report({"ERROR"}, "Text is required.")
            return {"CANCELLED"}

        space = getattr(context, "space_data", None)
        params = getattr(space, "params", None) if space else None
        directory = getattr(params, "directory", None) if params else None
        if directory is None:
            self.report({"ERROR"}, "Could not determine File Browser directory.")
            return {"CANCELLED"}
        if isinstance(directory, (bytes, bytearray)):
            try:
                directory = directory.decode("utf-8", errors="replace")
            except Exception:
                directory = str(directory)
        directory = str(directory)
        base_dir = Path(directory)
        if not base_dir.exists():
            self.report({"ERROR"}, f"Folder not found: {base_dir}")
            return {"CANCELLED"}

        files_map = _batch_dir_file_map(base_dir)
        if self.case_sensitive:
            matches = [
                path for _name_lower, path in files_map.items() if needle in path.name
            ]
        else:
            needle_l = needle.lower()
            matches = [
                path for name_lower, path in files_map.items() if needle_l in name_lower
            ]
        matches.sort(key=lambda p: p.name.lower())
        if not matches:
            self.report({"WARNING"}, "No matching files found.")
            return {"CANCELLED"}

        # Populate the invoking file-select operator's `files` collection so pressing OK
        # will return the full list (this is what Blender uses for multi-select).
        space = getattr(context, "space_data", None)
        active_op = getattr(space, "active_operator", None) if space else None
        if (
            active_op is not None
            and hasattr(active_op, "files")
            and hasattr(active_op, "directory")
        ):
            try:
                # Blender typically expects a trailing separator for `directory`.
                active_op.directory = str(base_dir) + (
                    "" if str(base_dir).endswith(("\\", "/")) else "\\"
                )
            except Exception:
                try:
                    active_op.directory = str(base_dir)
                except Exception:
                    pass
            try:
                active_op.files.clear()
                for path in matches:
                    entry = active_op.files.add()
                    entry.name = path.name
            except Exception:
                pass

        # Best-effort: reflect selection in the File Browser UI selection (not required
        # for OK to work, but helps the user see what's selected).
        selected = 0
        try:
            area = getattr(context, "area", None)
            region = getattr(context, "region", None)
            if not (area and getattr(area, "type", "") == "FILE_BROWSER"):
                area = None
            if area is None:
                for a in context.window.screen.areas:
                    if a.type == "FILE_BROWSER":
                        area = a
                        break
            if area is not None and (
                region is None or getattr(region, "type", "") != "WINDOW"
            ):
                for r in area.regions:
                    if r.type == "WINDOW":
                        region = r
                        break
            if area is not None and region is not None:
                with context.temp_override(area=area, region=region):
                    if not self.extend_selection:
                        try:
                            bpy.ops.file.select_all(action="DESELECT")
                        except Exception:
                            pass
                    for path in matches:
                        try:
                            bpy.ops.file.select(filepath=str(path), extend=True)
                        except Exception:
                            try:
                                bpy.ops.file.select(filepath=str(path))
                            except Exception:
                                continue
                        selected += 1
        except Exception:
            selected = 0

        # Try to reflect the selection in the File Browser filename field.
        # (This does not reliably drive multi-select by itself.)
        try:
            joined = ";".join([p.name for p in matches])
            if len(joined) > 4096:
                joined = matches[-1].name
            params.filename = joined
        except Exception:
            pass

        # If UI selection didn't update, still report how many will be returned on OK.
        self.report({"INFO"}, f"Selected {selected or len(matches)} file(s).")
        return {"FINISHED"}


def _batch_first_image_from_lists(lists) -> bpy.types.Image | None:
    for items in lists:
        for it in items:
            img = getattr(it, "image", None)
            if img is not None:
                return img
    return None


def _get_active_image_in_image_editor(context):
    for space in _iter_image_editor_spaces(context):
        return getattr(space, "image", None)
    return None


def _batch_remove_output_image(blend_data, image) -> None:
    if image is None:
        return
    try:
        blend_data.images.remove(image, do_unlink=True)
    except Exception:
        pass


_BATCH_ALLOWED_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".tif",
    ".tiff",
    ".bmp",
    ".exr",
    ".hdr",
)


def _batch_dir_file_map(directory: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    try:
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in _BATCH_ALLOWED_EXTS:
                continue
            out[entry.name.lower()] = entry
    except Exception:
        return {}
    return out


def _apply_suffix_case(template_suffix: str, sample_suffix: str) -> str:
    tmpl = template_suffix or ""
    if not tmpl:
        return tmpl
    if not sample_suffix:
        return tmpl
    sample = sample_suffix
    if sample.startswith("_") and tmpl.startswith("_"):
        sample = sample[1:]
        prefix = "_"
        tmpl_body = tmpl[1:]
    else:
        prefix = ""
        tmpl_body = tmpl
    if sample.isupper():
        return prefix + tmpl_body.upper()
    if sample.islower():
        return prefix + tmpl_body.lower()
    if sample[:1].isupper() and sample[1:].islower():
        return prefix + tmpl_body[:1].upper() + tmpl_body[1:].lower()
    return tmpl


def _strip_suffix_case_insensitive(stem: str, suffix: str) -> str:
    if not suffix:
        return stem
    if stem.lower().endswith(suffix.lower()):
        return stem[: -len(suffix)]
    return stem


def _detect_rgba_suffixes_for_channel(stem: str, channel: str) -> dict[str, str] | None:
    families = {
        "letters": {"r": "r", "g": "g", "b": "b", "a": "a"},
        "words": {"r": "red", "g": "green", "b": "blue", "a": "alpha"},
    }
    for family_name, mapping in families.items():
        token = "_" + mapping[channel]
        if stem.lower().endswith(token.lower()):
            sample = stem[-len(token) :]
            result: dict[str, str] = {}
            for ch, word in mapping.items():
                result[ch] = _apply_suffix_case("_" + word, sample)
            return result
    return None


def _detect_single_suffixes(stem: str, active_kind: str) -> tuple[str, str] | None:
    alpha_candidates = ["_a", "_alpha"]
    rgb_candidates = ["_rgb", "_color"]
    if active_kind == "alpha":
        for token in alpha_candidates:
            if stem.lower().endswith(token.lower()):
                sample = stem[-len(token) :]
                alpha = sample
                rgb = _apply_suffix_case("_rgb", sample)
                return rgb, alpha
        return None
    for token in rgb_candidates:
        if stem.lower().endswith(token.lower()):
            sample = stem[-len(token) :]
            rgb = sample
            alpha = _apply_suffix_case("_alpha", sample)
            return rgb, alpha
    return None


def _detect_rgba_suffixes_any(stem: str) -> dict[str, str] | None:
    families = [
        {"r": "r", "g": "g", "b": "b", "a": "a"},
        {"r": "red", "g": "green", "b": "blue", "a": "alpha"},
    ]
    for mapping in families:
        for channel, word in mapping.items():
            token = "_" + word
            if stem.lower().endswith(token.lower()):
                sample = stem[-len(token) :]
                result: dict[str, str] = {}
                for ch, w in mapping.items():
                    result[ch] = _apply_suffix_case("_" + w, sample)
                return result
    return None


def _is_known_rgba_suffix_set(suffixes: dict[str, str]) -> bool:
    s = {k: (suffixes.get(k, "") or "").strip().lower() for k in ("r", "g", "b", "a")}
    if all(s.values()) and s == {"r": "_r", "g": "_g", "b": "_b", "a": "_a"}:
        return True
    if all(s.values()) and s == {
        "r": "_red",
        "g": "_green",
        "b": "_blue",
        "a": "_alpha",
    }:
        return True
    return False


def _is_known_single_suffix_set(rgb_suffix: str, alpha_suffix: str) -> bool:
    rgb = (rgb_suffix or "").strip().lower()
    alpha = (alpha_suffix or "").strip().lower()
    if rgb in {"", "_rgb", "_color"} and alpha in {"", "_a", "_alpha"}:
        return True
    return False


def _batch_get_dest_dir(props, *, dest_attr: str, folder_name: str, lists) -> Path:
    dest = (getattr(props, dest_attr, "") or "").strip()
    if dest:
        return Path(bpy.path.abspath(dest))
    first = _batch_first_image_from_lists(lists)
    base = (
        _batch_default_dir_for_image(first, folder_name)
        if first
        else Path(bpy.app.tempdir or ".") / folder_name
    )
    return base


def _batch_maybe_set_dest_dir(
    props, *, dest_attr: str, folder_name: str, image
) -> None:
    is_custom_attr = f"{dest_attr}_is_custom"
    is_custom = bool(getattr(props, is_custom_attr, False))
    current = (getattr(props, dest_attr, "") or "").strip()
    if is_custom and current:
        return
    if current:
        return
    if image is None:
        return
    dest_dir = _batch_default_dir_for_image(image, folder_name)
    try:
        setattr(props, dest_attr, str(dest_dir))
        setattr(props, is_custom_attr, False)
    except Exception:
        pass


class ChannelPackerOTBatchPackSingle(bpy.types.Operator):
    """Batch pack using RGB + Alpha lists"""

    bl_idname = "channelpacker.batch_pack_single"
    bl_label = "Process Batch Pack (RGB + Alpha)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        rgb_items = props.batch_single_rgb_items
        alpha_items = props.batch_single_alpha_items
        if len(rgb_items) != len(alpha_items) or len(rgb_items) == 0:
            self.report(
                {"ERROR"},
                "RGB List and Alpha List must have the same number of items.",
            )
            return {"CANCELLED"}

        dest_dir = _batch_get_dest_dir(
            props,
            dest_attr="batch_dest_dir_single",
            folder_name="Packed_Batch",
            lists=[rgb_items, alpha_items],
        )

        active_before = _get_active_image_in_image_editor(context)
        keep_outputs = bool(getattr(props, "batch_keep_output_images_single", False))
        prev = {
            "mode": props.mode,
            "rgb_image": props.rgb_image,
            "alpha_image": props.alpha_image,
            "save_path_single": props.save_path_single,
            "save_path_single_is_custom": props.save_path_single_is_custom,
        }
        processed = 0
        out_paths: list[Path] = []
        try:
            props.mode = "SINGLE"
            for i in range(len(rgb_items)):
                rgb = rgb_items[i].image
                alpha = alpha_items[i].image
                if rgb is None:
                    self.report({"ERROR"}, f"RGB image missing at index {i + 1}.")
                    return {"CANCELLED"}

                props.rgb_image = rgb
                props.alpha_image = alpha
                stem = _batch_base_stem_for_image(rgb)
                ext = _batch_ext_for_image(rgb)
                out_path = dest_dir / f"{stem}{ext}"
                out_paths.append(out_path)
                props.save_path_single = str(out_path)
                props.save_path_single_is_custom = True

                result = bpy.ops.channelpacker.pack_channels()
                if "CANCELLED" in result:
                    return {"CANCELLED"}
                processed += 1
        finally:
            props.mode = prev["mode"]
            props.rgb_image = prev["rgb_image"]
            props.alpha_image = prev["alpha_image"]
            props.save_path_single = prev["save_path_single"]
            props.save_path_single_is_custom = prev["save_path_single_is_custom"]
            _set_active_image_in_image_editor(context, active_before)
            if not keep_outputs and out_paths:
                try:
                    props.result_image = None
                except Exception:
                    pass
                for path in out_paths:
                    out_img = _find_image_by_abspath(context.blend_data, str(path))
                    _batch_remove_output_image(context.blend_data, out_img)

        self.report({"INFO"}, f"Batch packed {processed} image(s).")
        return {"FINISHED"}


class ChannelPackerOTBatchPackMulti(bpy.types.Operator):
    """Batch pack using R/G/B/A lists"""

    bl_idname = "channelpacker.batch_pack_multi"
    bl_label = "Process Batch Pack (R+G+B+A)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        r_items = props.batch_multi_r_items
        g_items = props.batch_multi_g_items
        b_items = props.batch_multi_b_items
        a_items = props.batch_multi_a_items

        counts = {len(r_items), len(g_items), len(b_items), len(a_items)}
        if len(counts) != 1 or len(r_items) == 0:
            self.report(
                {"ERROR"},
                "R, G, B, and A lists must have the same number of items.",
            )
            return {"CANCELLED"}

        dest_dir = _batch_get_dest_dir(
            props,
            dest_attr="batch_dest_dir_multi",
            folder_name="Packed_Batch",
            lists=[r_items, g_items, b_items, a_items],
        )

        active_before = _get_active_image_in_image_editor(context)
        keep_outputs = bool(getattr(props, "batch_keep_output_images_multi", False))
        prev = {
            "mode": props.mode,
            "r_image": props.r_image,
            "g_image": props.g_image,
            "b_image": props.b_image,
            "a_image": props.a_image,
            "save_path_multi": props.save_path_multi,
            "save_path_multi_is_custom": props.save_path_multi_is_custom,
        }
        processed = 0
        out_paths: list[Path] = []
        try:
            props.mode = "MULTI"
            for i in range(len(r_items)):
                r_img = r_items[i].image
                g_img = g_items[i].image
                b_img = b_items[i].image
                a_img = a_items[i].image
                if all(img is None for img in (r_img, g_img, b_img, a_img)):
                    self.report({"ERROR"}, f"No channel images at index {i + 1}.")
                    return {"CANCELLED"}

                props.r_image = r_img
                props.g_image = g_img
                props.b_image = b_img
                props.a_image = a_img

                base_img = r_img or g_img or b_img or a_img
                stem = _batch_base_stem_for_image(base_img)
                ext = _batch_ext_for_image(base_img)
                out_path = dest_dir / f"{stem}{ext}"
                out_paths.append(out_path)
                props.save_path_multi = str(out_path)
                props.save_path_multi_is_custom = True

                result = bpy.ops.channelpacker.pack_channels()
                if "CANCELLED" in result:
                    return {"CANCELLED"}
                processed += 1
        finally:
            props.mode = prev["mode"]
            props.r_image = prev["r_image"]
            props.g_image = prev["g_image"]
            props.b_image = prev["b_image"]
            props.a_image = prev["a_image"]
            props.save_path_multi = prev["save_path_multi"]
            props.save_path_multi_is_custom = prev["save_path_multi_is_custom"]
            _set_active_image_in_image_editor(context, active_before)
            if not keep_outputs and out_paths:
                try:
                    props.result_image = None
                except Exception:
                    pass
                for path in out_paths:
                    out_img = _find_image_by_abspath(context.blend_data, str(path))
                    _batch_remove_output_image(context.blend_data, out_img)

        self.report({"INFO"}, f"Batch packed {processed} image(s).")
        return {"FINISHED"}


class ChannelPackerOTBatchSplit(bpy.types.Operator):
    """Batch split using the Split list"""

    bl_idname = "channelpacker.batch_split"
    bl_label = "Process Batch Split"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        items = props.batch_split_items
        if len(items) == 0:
            return {"CANCELLED"}

        dest_dir = _batch_get_dest_dir(
            props,
            dest_attr="batch_dest_dir_split",
            folder_name="Split_Batch",
            lists=[items],
        )

        active_before = _get_active_image_in_image_editor(context)
        keep_outputs = bool(getattr(props, "batch_keep_output_images_split", False))
        prev = {
            "mode": props.mode,
            "split_image": props.split_image,
            "save_path_split": props.save_path_split,
            "save_path_split_is_custom": props.save_path_split_is_custom,
            "split_name_r": props.split_name_r,
            "split_name_g": props.split_name_g,
            "split_name_b": props.split_name_b,
            "split_name_a": props.split_name_a,
            "split_name_r_is_custom": props.split_name_r_is_custom,
            "split_name_g_is_custom": props.split_name_g_is_custom,
            "split_name_b_is_custom": props.split_name_b_is_custom,
            "split_name_a_is_custom": props.split_name_a_is_custom,
        }
        processed = 0
        out_paths: list[Path] = []
        try:
            props.mode = "SPLIT"
            for i in range(len(items)):
                img = items[i].image
                if img is None:
                    self.report({"ERROR"}, f"Split image missing at index {i + 1}.")
                    return {"CANCELLED"}

                props.split_image = img
                base = _batch_base_stem_for_image(img)
                ext = _batch_ext_for_image(img)
                props.save_path_split = str(dest_dir / f"{base}{ext}")
                props.save_path_split_is_custom = True
                # Force per-image channel suffixes.
                props.split_name_r = f"{base}_red"
                props.split_name_g = f"{base}_green"
                props.split_name_b = f"{base}_blue"
                props.split_name_a = f"{base}_alpha"
                props.split_name_r_is_custom = True
                props.split_name_g_is_custom = True
                props.split_name_b_is_custom = True
                props.split_name_a_is_custom = True

                result = bpy.ops.channelpacker.split_channels()
                if "CANCELLED" in result:
                    return {"CANCELLED"}
                out_paths.extend(
                    [
                        dest_dir / f"{base}_red{ext}",
                        dest_dir / f"{base}_green{ext}",
                        dest_dir / f"{base}_blue{ext}",
                        dest_dir / f"{base}_alpha{ext}",
                    ]
                )
                processed += 1
        finally:
            props.mode = prev["mode"]
            props.split_image = prev["split_image"]
            props.save_path_split = prev["save_path_split"]
            props.save_path_split_is_custom = prev["save_path_split_is_custom"]
            props.split_name_r = prev["split_name_r"]
            props.split_name_g = prev["split_name_g"]
            props.split_name_b = prev["split_name_b"]
            props.split_name_a = prev["split_name_a"]
            props.split_name_r_is_custom = prev["split_name_r_is_custom"]
            props.split_name_g_is_custom = prev["split_name_g_is_custom"]
            props.split_name_b_is_custom = prev["split_name_b_is_custom"]
            props.split_name_a_is_custom = prev["split_name_a_is_custom"]
            _set_active_image_in_image_editor(context, active_before)
            if not keep_outputs and out_paths:
                try:
                    props.result_image = None
                except Exception:
                    pass
                for path in out_paths:
                    out_img = _find_image_by_abspath(context.blend_data, str(path))
                    if out_img is None:
                        out_img = context.blend_data.images.get(path.stem)
                    _batch_remove_output_image(context.blend_data, out_img)

        self.report({"INFO"}, f"Batch split {processed} image(s).")
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Panel: User Interface in the Image Editor's N-panel
# -----------------------------------------------------------------------------
class ChannelPackerPTPanel(bpy.types.Panel):
    """Panel for channel packing in the Image Editor"""

    bl_label = "Beyond Channel Packer"
    bl_idname = "CHANNELPACKER_PT_panel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Channel Packer"

    def draw(self, context):
        layout = self.layout
        props = context.scene.beyond_channel_packer
        if _needs_default_save_paths(props):
            _schedule_default_save_paths(context.scene)

        def draw_slot(parent, prop_name: str, label: str, label_factor: float):
            slot = parent.box()
            row = slot.row(align=True)
            split = row.split(factor=label_factor, align=True)
            split.label(text=label)
            right = split.row(align=True)
            right.prop(props, prop_name, text="")
            op = right.operator(
                "channelpacker.load_image",
                text="",
                icon="FILE_FOLDER",
            )
            op.target_prop = prop_name

        mode_box = layout.box()
        mode_box.label(text="Mode", icon="OPTIONS")
        mode_box.row(align=True).prop(props, "mode", expand=True)

        layout.separator()
        layout.separator()

        if props.mode != "BATCH":
            sources_box = layout.box()
            sources_box.label(text="Source Images", icon="IMAGE_DATA")
            if props.mode == "SINGLE":
                draw_slot(sources_box, "rgb_image", "RGB", 0.32)
                draw_slot(sources_box, "alpha_image", "Alpha (Optional)", 0.32)
            elif props.mode == "MULTI":
                grid = sources_box.grid_flow(
                    columns=2, even_columns=True, even_rows=False, align=True
                )
                draw_slot(grid, "r_image", "Red", 0.22)
                draw_slot(grid, "g_image", "Green", 0.22)
                draw_slot(grid, "b_image", "Blue", 0.22)
                draw_slot(grid, "a_image", "Alpha", 0.22)
            else:
                draw_slot(sources_box, "split_image", "Source", 0.32)

            layout.separator()
            layout.separator()

            result_box = layout.box()
            result_box.label(text="Result", icon="RENDER_RESULT")
            save_row = result_box.row(align=True)
            if props.mode == "SINGLE":
                save_row.prop(props, "save_path_single", text="Save To")
            elif props.mode == "MULTI":
                save_row.prop(props, "save_path_multi", text="Save To")
            else:
                save_row.prop(props, "save_path_split", text="Save To")
                names_box = result_box.box()
                names_box.label(text="Output Names")
                has_alpha = False
                if props.split_image is not None:
                    try:
                        ch = int(getattr(props.split_image, "channels", 0) or 0)
                    except Exception:
                        ch = 0
                    if ch == 3:
                        has_alpha = False
                    else:
                        has_alpha = True
                grid = names_box.grid_flow(
                    columns=2,
                    even_columns=True,
                    even_rows=False,
                    align=True,
                )
                grid.prop(props, "split_name_r")
                grid.prop(props, "split_name_g")
                grid.prop(props, "split_name_b")
                if has_alpha:
                    grid.prop(props, "split_name_a")

            if props.mode == "SPLIT":
                result_box.operator("channelpacker.split_channels", icon="FILE_BLEND")
            else:
                result_box.operator("channelpacker.pack_channels", icon="FILE_BLEND")
            result_box.prop(
                props,
                "override_img_if_exists",
                text="Override Img if Exists",
            )
        else:

            def draw_batch_list(parent, label: str, collection_prop: str):
                box = parent.box()
                header = box.row(align=True)
                header.label(text=label)
                buttons = header.row(align=True)
                op = buttons.operator(
                    "channelpacker.batch_list_add", text="", icon="FILE_FOLDER"
                )
                op.collection_prop = collection_prop
                op = buttons.operator(
                    "channelpacker.batch_list_remove", text="", icon="REMOVE"
                )
                op.collection_prop = collection_prop
                op = buttons.operator(
                    "channelpacker.batch_list_clear", text="", icon="TRASH"
                )
                op.collection_prop = collection_prop
                box.template_list(
                    "CHANNELPACKER_UL_batch_images",
                    "",
                    props,
                    collection_prop,
                    props,
                    _BATCH_LIST_PROPS[collection_prop],
                    rows=4,
                )

            packed_box = layout.box()
            packed_header = packed_box.row(align=True)
            packed_header.prop(
                props,
                "batch_show_single",
                text="",
                icon="TRIA_DOWN" if props.batch_show_single else "TRIA_RIGHT",
                emboss=False,
            )
            packed_header.label(text="Batch: RGB + Alpha", icon="IMAGE_RGB_ALPHA")
            if props.batch_show_single:
                grid = packed_box.grid_flow(
                    columns=2, even_columns=True, even_rows=False, align=True
                )
                draw_batch_list(grid, "RGB List", "batch_single_rgb_items")
                draw_batch_list(grid, "Alpha List", "batch_single_alpha_items")
                packed_box.separator()
                packed_box.prop(
                    props, "batch_dest_dir_single", text="Destination Folder"
                )
                if not (props.batch_dest_dir_single or "").strip():
                    suggested = _batch_get_dest_dir(
                        props,
                        dest_attr="batch_dest_dir_single",
                        folder_name="Packed_Batch",
                        lists=[
                            props.batch_single_rgb_items,
                            props.batch_single_alpha_items,
                        ],
                    )
                    hint = packed_box.column()
                    hint.enabled = False
                    hint.label(text=f"Default: {suggested}")
                can_run_single = (
                    len(props.batch_single_rgb_items)
                    == len(props.batch_single_alpha_items)
                    and len(props.batch_single_rgb_items) > 0
                )
                run_row = packed_box.row()
                run_row.enabled = can_run_single
                run_row.operator("channelpacker.batch_pack_single", icon="FILE_BLEND")
                packed_box.prop(
                    props,
                    "batch_keep_output_images_single",
                    text="Keep output images in Blender",
                )
                if len(props.batch_single_rgb_items) != len(
                    props.batch_single_alpha_items
                ) and (
                    len(props.batch_single_rgb_items)
                    or len(props.batch_single_alpha_items)
                ):
                    warn = packed_box.column()
                    warn.alert = True
                    warn.label(
                        text="! RGB List and Alpha List must have the same number of items!",
                        icon="ERROR",
                    )

            packed4_box = layout.box()
            packed4_header = packed4_box.row(align=True)
            packed4_header.prop(
                props,
                "batch_show_multi",
                text="",
                icon="TRIA_DOWN" if props.batch_show_multi else "TRIA_RIGHT",
                emboss=False,
            )
            packed4_header.label(text="Batch: 4-Channel (R+G+B+A)", icon="IMAGE_DATA")
            if props.batch_show_multi:
                grid = packed4_box.grid_flow(
                    columns=2, even_columns=True, even_rows=False, align=True
                )
                draw_batch_list(grid, "R List", "batch_multi_r_items")
                draw_batch_list(grid, "G List", "batch_multi_g_items")
                draw_batch_list(grid, "B List", "batch_multi_b_items")
                draw_batch_list(grid, "A List", "batch_multi_a_items")
                packed4_box.separator()
                packed4_box.prop(
                    props, "batch_dest_dir_multi", text="Destination Folder"
                )
                if not (props.batch_dest_dir_multi or "").strip():
                    suggested = _batch_get_dest_dir(
                        props,
                        dest_attr="batch_dest_dir_multi",
                        folder_name="Packed_Batch",
                        lists=[
                            props.batch_multi_r_items,
                            props.batch_multi_g_items,
                            props.batch_multi_b_items,
                            props.batch_multi_a_items,
                        ],
                    )
                    hint = packed4_box.column()
                    hint.enabled = False
                    hint.label(text=f"Default: {suggested}")
                counts = {
                    len(props.batch_multi_r_items),
                    len(props.batch_multi_g_items),
                    len(props.batch_multi_b_items),
                    len(props.batch_multi_a_items),
                }
                can_run_multi = len(counts) == 1 and len(props.batch_multi_r_items) > 0
                run_row = packed4_box.row()
                run_row.enabled = can_run_multi
                run_row.operator("channelpacker.batch_pack_multi", icon="FILE_BLEND")
                packed4_box.prop(
                    props,
                    "batch_keep_output_images_multi",
                    text="Keep output images in Blender",
                )
                if len(counts) != 1 and sum(counts) > 0:
                    warn = packed4_box.column()
                    warn.alert = True
                    warn.label(
                        text="! R, G, B, and A lists must have the same number of items!",
                        icon="ERROR",
                    )

            split_box = layout.box()
            split_header = split_box.row(align=True)
            split_header.prop(
                props,
                "batch_show_split",
                text="",
                icon="TRIA_DOWN" if props.batch_show_split else "TRIA_RIGHT",
                emboss=False,
            )
            split_header.label(text="Batch: Split Channels", icon="MOD_EXPLODE")
            if props.batch_show_split:
                draw_batch_list(split_box, "Split List", "batch_split_items")
                split_box.separator()
                split_box.prop(props, "batch_dest_dir_split", text="Destination Folder")
                if not (props.batch_dest_dir_split or "").strip():
                    suggested = _batch_get_dest_dir(
                        props,
                        dest_attr="batch_dest_dir_split",
                        folder_name="Split_Batch",
                        lists=[props.batch_split_items],
                    )
                    hint = split_box.column()
                    hint.enabled = False
                    hint.label(text=f"Default: {suggested}")
                run_row = split_box.row()
                run_row.enabled = len(props.batch_split_items) > 0
                run_row.operator("channelpacker.batch_split", icon="FILE_BLEND")
                split_box.prop(
                    props,
                    "batch_keep_output_images_split",
                    text="Keep output images in Blender",
                )

        # Result is shown by setting the active Image Editor image.

        layout.separator()
        layout.separator()

        adv_box = layout.box()
        header = adv_box.row(align=True)
        header.prop(
            props,
            "show_advanced",
            text="Advanced Options",
            icon="TRIA_DOWN" if props.show_advanced else "TRIA_RIGHT",
            emboss=False,
        )
        if props.show_advanced:
            options_col = adv_box.column(align=True)
            options_col.prop(
                props,
                "auto_scale_to_largest",
                text="Auto Scale to Largest",
            )
            sub = options_col.row(align=True)
            sub.enabled = props.auto_scale_to_largest
            sub.prop(props, "aspect_ratio_tolerance", text="Aspect Tolerance")

            adjust_col = adv_box.column(align=True)
            adjust_col.enabled = props.mode == "MULTI"
            adjust_col.prop(props, "multi_gamma")
            adjust_col.prop(props, "multi_contrast")
            adjust_col.prop(props, "multi_brightness")


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
classes = (
    BeyondChannelPackerBatchImageItem,
    BeyondChannelPackerProperties,
    ChannelPackerOTPackChannels,
    ChannelPackerOTSplitChannels,
    ChannelPackerOTLoadImage,
    ChannelPackerULBatchImages,
    ChannelPackerOTBatchListAdd,
    ChannelPackerOTBatchListRemove,
    ChannelPackerOTBatchListClear,
    ChannelPackerOTFileBrowserSelectMatching,
    ChannelPackerOTBatchPackSingle,
    ChannelPackerOTBatchPackMulti,
    ChannelPackerOTBatchSplit,
    ChannelPackerPTPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # Be tolerant of reloads where the property may already exist.
    if hasattr(bpy.types.Scene, "beyond_channel_packer"):
        del bpy.types.Scene.beyond_channel_packer
    bpy.types.Scene.beyond_channel_packer = bpy.props.PointerProperty(
        type=BeyondChannelPackerProperties
    )


def unregister():
    # Be tolerant of partial registrations or reloads.
    if hasattr(bpy.types.Scene, "beyond_channel_packer"):
        del bpy.types.Scene.beyond_channel_packer

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
