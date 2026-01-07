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

from pathlib import Path

import bpy


_MULTI_SRGB_GAMMA = 1.2
_MULTI_SRGB_CONTRAST = 0.985
_MULTI_SRGB_BRIGHTNESS = 0.0

bl_info = {
    "name": "BeyondChannelPacker",
    "author": "Beyond Dev (Tyler Walker)",
    "version": (0, 1, 5),
    "blender": (4, 2, 0),
    "description": (
        "Channel pack images in the Image Editor (choose RGB+Alpha or 4 separate "
        "channels)."
    ),
    "category": "Image",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_DEFAULT_SAVE_PATH_SCENES_PENDING: set[str] = set()


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
    base_target = base if base else _autosave_target_for_sources(sources)

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


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
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
                if not fp:
                    return
                fp = bpy.path.abspath(fp)
                src_path = Path(fp)
                if src_path.suffix:
                    packed_path = src_path.with_name(
                        f"{src_path.stem}_packed{src_path.suffix}"
                    )
                else:
                    packed_path = src_path.with_name(f"{src_path.name}_packed.png")

                if prop_name == "rgb_image":
                    if not getattr(self, "save_path_single_is_custom", False):
                        self.save_path_single = str(packed_path)
                        self.save_path_single_is_custom = False
                if prop_name == "r_image":
                    if not getattr(self, "save_path_multi_is_custom", False):
                        self.save_path_multi = str(packed_path)
                        self.save_path_multi_is_custom = False
            except Exception:
                pass

    return update


def _default_save_path_for_image(image) -> str | None:
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
        return str(src_path.with_name(f"{src_path.stem}_packed{src_path.suffix}"))
    return str(src_path.with_name(f"{src_path.name}_packed.png"))


def _ensure_default_save_paths(props) -> None:
    if not (getattr(props, "save_path_single", "") or "").strip() and props.rgb_image:
        suggested = _default_save_path_for_image(props.rgb_image)
        if suggested:
            props.save_path_single = suggested
            props.save_path_single_is_custom = False
    if not (getattr(props, "save_path_multi", "") or "").strip() and props.r_image:
        suggested = _default_save_path_for_image(props.r_image)
        if suggested:
            props.save_path_multi = suggested
            props.save_path_multi_is_custom = False


# -----------------------------------------------------------------------------
# Property Group: Stores all settings for channel packing.
# -----------------------------------------------------------------------------
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
                "Separate Channels",
                "Pack four separate channel images (R, G, B, A)",
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

    debug_info: bpy.props.StringProperty(
        name="Debug Info",
        description="Last debug info from packing",
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
            "(e.g. 0.02 = 2%)"
        ),
        default=0.02,
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
        props.last_saved_path = ""
        props.debug_info = ""
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
            target_img = max(
                selected_images, key=lambda img: img.size[0] * img.size[1]
            )
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

        def ensure_image_pixels_loaded(img) -> None:
            if img is None:
                return
            try:
                if getattr(img, "source", "") == "FILE" and not getattr(
                    img, "has_data", True
                ):
                    img.reload()
            except RuntimeError:
                pass
            try:
                _ = img.pixels[0]
            except Exception:
                pass

        def _resample_nearest(arr, src_w: int, src_h: int):
            if src_w == width and src_h == height:
                return arr
            y_idx = (np.arange(height) * (src_h / height)).astype(np.int32)
            x_idx = (np.arange(width) * (src_w / width)).astype(np.int32)
            return arr[y_idx[:, None], x_idx[None, :], :]

        def _linear_to_srgb(x):
            x = np.clip(x, 0.0, 1.0)
            return np.where(
                x <= 0.0031308,
                x * 12.92,
                1.055 * np.power(x, 1.0 / 2.4) - 0.055,
            )

        def read_pixels_1d(img, *, treat_as_data: bool):
            ensure_image_pixels_loaded(img)

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

        def _debug_image_stats(label: str, img, data):
            if data is None:
                print(f"[BeyondChannelPacker] {label}: <no data>")
                return
            try:
                fp = getattr(img, "filepath", "")
            except Exception:
                fp = ""
            print(
                "[BeyondChannelPacker]",
                label,
                "min/max",
                float(np.min(data)),
                float(np.max(data)),
                "img",
                getattr(img, "name", "<none>"),
                fp,
            )

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
                    return None, None
                try:
                    cs = img.colorspace_settings.name
                except Exception:
                    cs = None
                arr = get_pixels_array(img, treat_as_data=False)
                if arr is None:
                    return None, cs
                data = extract_greyscale(arr, prefer_alpha=True)
                return data, cs

            r_data, r_cs = read_channel(props.r_image)
            g_data, g_cs = read_channel(props.g_image)
            b_data, b_cs = read_channel(props.b_image)
            a_data, a_cs = read_channel(props.a_image)

            output[:, :, 0] = r_data if r_data is not None else 0.0
            output[:, :, 1] = g_data if g_data is not None else 0.0
            output[:, :, 2] = b_data if b_data is not None else 0.0
            output[:, :, 3] = a_data if a_data is not None else 1.0

            if float(output[:, :, 0:3].max()) <= 1e-6:
                _debug_image_stats("MULTI R", props.r_image, r_data)
                _debug_image_stats("MULTI G", props.g_image, g_data)
                _debug_image_stats("MULTI B", props.b_image, b_data)
                _debug_image_stats("MULTI A", props.a_image, a_data)

            def _mm(x):
                if x is None:
                    return "None"
                return f"{float(np.min(x)):.4f},{float(np.max(x)):.4f}"

            props.debug_info = (
                f"R[{_mm(r_data)}] G[{_mm(g_data)}] B[{_mm(b_data)}] "
                f"A[{_mm(a_data)}]"
            )
            props.debug_info = (
                f"{props.debug_info} "
                f"INCS[{r_cs},{g_cs},{b_cs},{a_cs}]"
            )

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

            props.debug_info = (
                f"{props.debug_info} "
                f"ADJ[G{multi_gamma:.4f},C{multi_contrast:.4f},B{multi_brightness:.4f}]"
            )

        flat_pixels = output.astype(np.float32, copy=False).reshape(-1).tolist()
        out_mins = tuple(float(v) for v in np.min(output, axis=(0, 1)))
        out_maxs = tuple(float(v) for v in np.max(output, axis=(0, 1)))

        def _readback_minmax(img):
            try:
                px_len = len(img.pixels)
            except Exception:
                return None
            if px_len <= 0:
                return None
            rb = np.empty(px_len, dtype=np.float32)
            try:
                img.pixels.foreach_get(rb)
            except Exception:
                try:
                    rb = np.array(img.pixels[:], dtype=np.float32)
                except Exception:
                    return None
            if rb.size < 4:
                return None
            rb = rb.reshape((-1, 4))
            mins = np.min(rb, axis=0)
            maxs = np.max(rb, axis=0)
            return tuple(float(v) for v in mins), tuple(float(v) for v in maxs)

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
        # Create a new image datablock for the packed result.
        # -------------------------------------------------------------------------
            source_for_colorspace = None
            if mode == "SINGLE":
                source_for_colorspace = props.rgb_image
            else:
                source_for_colorspace = (
                    props.r_image
                    or props.g_image
                    or props.b_image
                    or props.a_image
                )

            sources = (
                [props.rgb_image, props.alpha_image]
                if mode == "SINGLE"
                else [props.r_image, props.g_image, props.b_image, props.a_image]
            )
            base_target = _autosave_target_for_sources(sources)
            base_target_path = Path(bpy.path.abspath(base_target))
            file_format, ext = _autosave_format_for_path(str(base_target_path))
            wants_float = ext in {".exr", ".hdr"}
            for src in sources:
                if src is None:
                    continue
                if bool(getattr(src, "is_float", False)):
                    wants_float = True
                    break
            if mode == "MULTI":
                wants_float = True

            base_img = None
            if source_for_colorspace is not None:
                try:
                    base_img = source_for_colorspace.copy()
                except Exception:
                    base_img = None

            if mode == "MULTI":
                base_img = None

                def _make_multi_file_backed_base():
                    for src in sources:
                        if src is None:
                            continue
                        filepath = (
                            getattr(src, "filepath_raw", "")
                            or getattr(src, "filepath", "")
                            or ""
                        ).strip()
                        if not filepath:
                            continue
                        filepath_abs = bpy.path.abspath(filepath)
                        if not filepath_abs:
                            continue
                        try:
                            img = context.blend_data.images.load(
                                filepath_abs,
                                check_existing=False,
                            )
                        except Exception:
                            continue
                        return img
                    return None

                base_img = _make_multi_file_backed_base()

                if base_img is None and target_img is not None:
                    try:
                        base_img = target_img.copy()
                    except Exception:
                        base_img = None

            if base_img is not None:
                try:
                    _ = base_img.pixels[0]
                except Exception:
                    pass
                try:
                    base_img.name = _unique_image_name(
                        context.blend_data,
                        "ChannelPacked",
                    )
                except Exception:
                    pass
                try:
                    base_img.use_fake_user = False
                except Exception:
                    pass
                if int(base_img.size[0]) != width or int(base_img.size[1]) != height:
                    if mode == "MULTI" and getattr(base_img, "source", "") != "FILE":
                        try:
                            context.blend_data.images.remove(base_img, do_unlink=True)
                        except Exception:
                            pass
                        base_img = None
                    else:
                        ensure_image_pixels_loaded(base_img)
                        try:
                            base_img.scale(width, height)
                        except Exception as exc:
                            self.report(
                                {"ERROR"},
                                f"Could not scale base image '{base_img.name}': {exc}",
                            )
                            return {"CANCELLED"}
            if base_img is None:
                result_img = context.blend_data.images.new(
                    name=_unique_image_name(context.blend_data, "ChannelPacked"),
                    width=width,
                    height=height,
                    alpha=True,
                    float_buffer=wants_float,
                )
            else:
                result_img = base_img

            try:
                result_img.use_view_as_render = False
            except Exception:
                pass
            try:
                result_img.alpha_mode = "STRAIGHT"
            except (AttributeError, TypeError):
                pass
            try:
                if mode == "MULTI":
                    result_img.alpha_mode = "STRAIGHT"
            except Exception:
                pass
            if source_for_colorspace is not None:
                try:
                    result_img.colorspace_settings.name = (
                        source_for_colorspace.colorspace_settings.name
                    )
                except (AttributeError, TypeError):
                    pass

            set_status = assign_pixels(result_img, flat_pixels)
            props.result_image = result_img

            rb = _readback_minmax(props.result_image)
            try:
                alpha_mode_dbg = props.result_image.alpha_mode
            except Exception:
                alpha_mode_dbg = "?"
            try:
                px_len_dbg = len(props.result_image.pixels)
            except Exception:
                px_len_dbg = -1
            try:
                is_float_dbg = bool(getattr(props.result_image, "is_float", False))
            except Exception:
                is_float_dbg = False
            buf_dbg = f"BUF[len={px_len_dbg},float={is_float_dbg}]"
            if rb is not None:
                rb_mins, rb_maxs = rb
                props.debug_info = (
                    f"{props.debug_info} OUT[{out_mins},{out_maxs}] "
                    f"WRITE[{rb_mins},{rb_maxs}] SET[{set_status}] {buf_dbg}"
                ).strip()
            else:
                props.debug_info = (
                    f"{props.debug_info} OUT[{out_mins},{out_maxs}] "
                    f"WRITE[?] SET[{set_status}] {buf_dbg}"
                ).strip()
            props.debug_info = (
                f"{props.debug_info} AM[{alpha_mode_dbg}]"
            ).strip()

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

            props.result_image.filepath_raw = str(out_path)
            try:
                props.result_image.file_format = file_format
            except (AttributeError, TypeError):
                pass
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                props.result_image.save()
                props.last_saved_path = str(out_path)
            except RuntimeError as exc:
                self.report(
                    {"WARNING"},
                    f"Packed, but could not auto-save result: {exc}",
                )

        _set_active_image_in_image_editor(context, props.result_image)

        if props.last_saved_path:
            self.report(
                {"INFO"},
                f"Channel packing complete. Saved: {props.last_saved_path}",
            )
        else:
            self.report({"INFO"}, "Channel packing complete.")
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


class ChannelPackerOTCopyDebug(bpy.types.Operator):
    bl_idname = "channelpacker.copy_debug"
    bl_label = "Copy Debug"
    bl_description = "Copy the debug info text to the clipboard"

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        text = props.debug_info or ""
        try:
            context.window_manager.clipboard = text
        except Exception:
            return {"CANCELLED"}
        return {"FINISHED"}


class ChannelPackerOTCopySavedPath(bpy.types.Operator):
    bl_idname = "channelpacker.copy_saved_path"
    bl_label = "Copy Saved Path"
    bl_description = "Copy the last saved path to the clipboard"

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        text = props.last_saved_path or ""
        try:
            context.window_manager.clipboard = text
        except Exception:
            return {"CANCELLED"}
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

        sources_box = layout.box()
        sources_box.label(text="Source Images", icon="IMAGE_DATA")
        if props.mode == "SINGLE":
            draw_slot(sources_box, "rgb_image", "RGB", 0.32)
            draw_slot(sources_box, "alpha_image", "Alpha (Optional)", 0.32)
        else:
            grid = sources_box.grid_flow(
                columns=2, even_columns=True, even_rows=False, align=True
            )
            draw_slot(grid, "r_image", "Red", 0.22)
            draw_slot(grid, "g_image", "Green", 0.22)
            draw_slot(grid, "b_image", "Blue", 0.22)
            draw_slot(grid, "a_image", "Alpha", 0.22)

        layout.separator()
        layout.separator()

        result_box = layout.box()
        result_box.label(text="Result", icon="RENDER_RESULT")
        save_row = result_box.row(align=True)
        if props.mode == "SINGLE":
            save_row.prop(props, "save_path_single", text="Save To")
        else:
            save_row.prop(props, "save_path_multi", text="Save To")
        result_box.operator("channelpacker.pack_channels", icon="FILE_BLEND")
        result_box.prop(
            props,
            "override_img_if_exists",
            text="Override Img if Exists",
        )

        if props.result_image:
            try:
                props.result_image.preview_ensure()
                icon_id = props.result_image.preview.icon_id
                if icon_id:
                    result_box.template_icon(icon_value=icon_id, scale=8.0)
                else:
                    result_box.label(text=props.result_image.name)
            except Exception:
                result_box.label(text=props.result_image.name)
        else:
            result_box.label(text="No packed result yet.")

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

            if props.debug_info:
                adv_box.separator()
                debug_row = adv_box.row(align=True)
                debug_row.prop(props, "debug_info", text="Debug")
                debug_row.operator(
                    "channelpacker.copy_debug",
                    text="",
                    icon="COPYDOWN",
                )


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
classes = (
    BeyondChannelPackerProperties,
    ChannelPackerOTPackChannels,
    ChannelPackerOTLoadImage,
    ChannelPackerOTCopyDebug,
    ChannelPackerOTCopySavedPath,
    ChannelPackerPTPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.beyond_channel_packer = bpy.props.PointerProperty(
        type=BeyondChannelPackerProperties
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.beyond_channel_packer


if __name__ == "__main__":
    register()
