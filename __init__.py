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


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    import binascii
    import struct

    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _write_rgba_png8(path: Path, *, width: int, height: int, rgba_u8_bytes: bytes):
    import struct
    import zlib

    expected = width * height * 4
    if width <= 0 or height <= 0 or len(rgba_u8_bytes) != expected:
        raise ValueError("Invalid PNG dimensions or pixel buffer length.")

    raw = bytearray((width * 4 + 1) * height)
    stride = width * 4 + 1
    for y in range(height):
        row_off = y * stride
        raw[row_off] = 0
        src_off = y * width * 4
        raw[row_off + 1 : row_off + 1 + width * 4] = rgba_u8_bytes[
            src_off : src_off + width * 4
        ]

    compressed = zlib.compress(bytes(raw), level=6)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(_png_chunk(b"IHDR", ihdr))
    png.extend(_png_chunk(b"IDAT", compressed))
    png.extend(_png_chunk(b"IEND", b""))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _u8_rgba_minmax(
    rgba_u8,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    import numpy as np

    flat = rgba_u8.reshape((-1, 4))
    mins = tuple(int(v) for v in flat.min(axis=0))
    maxs = tuple(int(v) for v in flat.max(axis=0))
    return mins, maxs

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

    return update


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
    # New property to override the original RGB image with the packed image.
    override_rgb: bpy.props.BoolProperty(
        name="Override RGB",
        description=(
            "Override the original RGB image with the packed image after packing"
        ),
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

        # If we're overriding the RGB image in SINGLE mode, prefer the RGB image's
        # dimensions so we can safely overwrite/save it back to disk.
        if mode == "SINGLE" and props.override_rgb and props.rgb_image is not None:
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
            if arr.shape[2] >= 3:
                rgb = np.maximum(np.maximum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
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
            arr_r = (
                get_pixels_array(props.r_image, treat_as_data=False)
                if props.r_image
                else None
            )
            arr_g = (
                get_pixels_array(props.g_image, treat_as_data=False)
                if props.g_image
                else None
            )
            arr_b = (
                get_pixels_array(props.b_image, treat_as_data=False)
                if props.b_image
                else None
            )
            arr_a = (
                get_pixels_array(props.a_image, treat_as_data=False)
                if props.a_image
                else None
            )

            r_data = extract_greyscale(arr_r, prefer_alpha=True)
            g_data = extract_greyscale(arr_g, prefer_alpha=True)
            b_data = extract_greyscale(arr_b, prefer_alpha=True)
            a_data = extract_greyscale(arr_a, prefer_alpha=True)

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
                img.alpha_mode = "STRAIGHT"
            except Exception:
                pass

            try:
                buf = array("f", pixels_list)
                img.pixels.foreach_set(buf)
            except Exception as exc:
                try:
                    img.pixels[:] = pixels_list
                except Exception as exc2:
                    return f"ERR:set {type(exc).__name__}/{type(exc2).__name__}"
                else:
                    try:
                        img.update()
                        img.update_tag()
                    except Exception:
                        pass
                    return "OK:slice"
            try:
                img.update()
            except Exception:
                pass
            try:
                img.update_tag()
            except Exception:
                pass
            return "OK:foreach"

        if mode == "MULTI":
            sources = [props.r_image, props.g_image, props.b_image, props.a_image]
            base_target = _autosave_target_for_sources(sources)
            base_target_path = Path(bpy.path.abspath(base_target))

            stem = base_target_path.stem or "ChannelPacked"
            out_dir = (
                base_target_path.parent if str(base_target_path.parent) else Path(".")
            )
            if stem == "ChannelPacked" or stem.endswith("_packed"):
                out_stem = stem
            else:
                out_stem = f"{stem}_packed"

            out_path = _unique_path(out_dir / f"{out_stem}.png")
            rgb_lin = np.clip(output[:, :, 0:3], 0.0, 1.0)
            rgb_srgb = np.where(
                rgb_lin <= 0.0031308,
                rgb_lin * 12.92,
                1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
            )
            encoded = output.copy()
            encoded[:, :, 0:3] = rgb_srgb
            rgba_u8 = np.clip(encoded * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            u8_mins, u8_maxs = _u8_rgba_minmax(rgba_u8)
            try:
                _write_rgba_png8(
                    out_path,
                    width=width,
                    height=height,
                    rgba_u8_bytes=rgba_u8.tobytes(order="C"),
                )
            except Exception as exc:
                self.report({"ERROR"}, f"Could not write PNG: {exc}")
                return {"CANCELLED"}

            try:
                result_img = context.blend_data.images.load(
                    str(out_path),
                    check_existing=False,
                )
            except Exception as exc:
                self.report({"ERROR"}, f"Could not load written PNG: {exc}")
                return {"CANCELLED"}

            try:
                result_img.alpha_mode = "STRAIGHT"
            except Exception:
                pass
            try:
                result_img.use_view_as_render = False
            except Exception:
                pass
            source_cs = None
            for src in sources:
                if src is None:
                    continue
                try:
                    source_cs = src.colorspace_settings.name
                except Exception:
                    source_cs = None
                if source_cs:
                    break
            if source_cs:
                try:
                    result_img.colorspace_settings.name = source_cs
                except Exception:
                    pass
            ensure_image_pixels_loaded(result_img)
            file_minmax = _readback_minmax(result_img)
            file_mins, file_maxs = (None, None)
            if file_minmax is not None:
                file_mins, file_maxs = file_minmax
            props.result_image = result_img
            props.last_saved_path = str(out_path)
            props.debug_info = (
                f"{props.debug_info} OUT[{out_mins},{out_maxs}] "
                f"U8[{u8_mins},{u8_maxs}] "
                f"FILE[{file_mins},{file_maxs}] WRITE[file] SET[PYPNG]"
            ).strip()
            _set_active_image_in_image_editor(context, props.result_image)
            self.report({"INFO"}, f"Channel packing complete. Saved: {out_path}")
            return {"FINISHED"}

        # -----------------------------------------------------------------------------
        # If override is enabled in SINGLE mode, write pixels back into the original
        # RGB image datablock (preserves filepath/colorspace/alpha settings) and
        # save directly to disk.
        # -----------------------------------------------------------------------------
        if mode == "SINGLE" and props.override_rgb:
            if props.rgb_image is None:
                self.report({"ERROR"}, "RGB Image is required.")
                return {"CANCELLED"}

            rgb_img = props.rgb_image
            set_status = assign_pixels(rgb_img, flat_pixels)
            props.result_image = rgb_img
            try:
                rgb_img.use_view_as_render = False
            except Exception:
                pass
            try:
                rgb_img.alpha_mode = "STRAIGHT"
            except (AttributeError, TypeError):
                pass

            filepath = (
                getattr(rgb_img, "filepath_raw", "")
                or getattr(rgb_img, "filepath", "")
                or ""
            ).strip()
            filepath_abs = bpy.path.abspath(filepath) if filepath else ""
            ext = Path(filepath_abs).suffix.lower() if filepath_abs else ""
            alpha_ok_exts = {".png", ".exr", ".tif", ".tiff", ".tga"}

            if filepath_abs and ext in alpha_ok_exts:
                file_format = _image_file_format_from_path(filepath_abs)
                if file_format is not None:
                    rgb_img.filepath_raw = filepath_abs
                    try:
                        rgb_img.file_format = file_format
                    except (AttributeError, TypeError):
                        pass
                    try:
                        rgb_img.save()
                        props.last_saved_path = filepath_abs
                    except RuntimeError as exc:
                        self.report(
                            {"WARNING"},
                            f"Packed, but could not overwrite source file: {exc}",
                        )

            if not props.last_saved_path:
                sources = [props.rgb_image, props.alpha_image]
                base_target = _autosave_target_for_sources(sources)
                base_target_path = Path(bpy.path.abspath(base_target))
                file_format, out_ext = _autosave_format_for_path(str(base_target_path))

                stem = base_target_path.stem or "ChannelPacked"
                out_dir = (
                    base_target_path.parent
                    if str(base_target_path.parent)
                    else Path(".")
                )
                if stem == "ChannelPacked" or stem.endswith("_packed"):
                    out_stem = stem
                else:
                    out_stem = f"{stem}_packed"
                out_path = _unique_path(out_dir / f"{out_stem}{out_ext}")

                orig_filepath_raw = getattr(rgb_img, "filepath_raw", "")
                orig_file_format = getattr(rgb_img, "file_format", None)
                rgb_img.filepath_raw = str(out_path)
                try:
                    rgb_img.file_format = file_format
                except (AttributeError, TypeError):
                    pass
                try:
                    rgb_img.save()
                    props.last_saved_path = str(out_path)
                except RuntimeError as exc:
                    self.report(
                        {"WARNING"},
                        f"Packed, but could not auto-save copy: {exc}",
                    )
                finally:
                    rgb_img.filepath_raw = orig_filepath_raw
                    if orig_file_format is not None:
                        try:
                            rgb_img.file_format = orig_file_format
                        except (AttributeError, TypeError):
                            pass
        else:
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

            base_img = None
            if source_for_colorspace is not None:
                try:
                    base_img = source_for_colorspace.copy()
                except Exception:
                    base_img = None

            if mode == "MULTI":
                tmp = None
                try:
                    tmp = context.blend_data.images.new(
                        name=_unique_image_name(context.blend_data, "BCP_TMP_RGBA"),
                        width=width,
                        height=height,
                        alpha=True,
                        float_buffer=wants_float,
                    )
                except Exception:
                    tmp = None

                if tmp is not None:
                    try:
                        _ = tmp.pixels[0]
                    except Exception:
                        pass
                    try:
                        base_img = tmp.copy()
                    except Exception:
                        base_img = tmp
                    try:
                        context.blend_data.images.remove(tmp, do_unlink=True)
                    except Exception:
                        pass

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
                    try:
                        base_img.scale(width, height)
                    except Exception as exc:
                        self.report(
                            {"ERROR"},
                            f"Could not scale base image '{base_img.name}': {exc}",
                        )
                        return {"CANCELLED"}
                result_img = base_img
            else:
                result_img = context.blend_data.images.new(
                    name=_unique_image_name(context.blend_data, "ChannelPacked"),
                    width=width,
                    height=height,
                    alpha=True,
                    float_buffer=wants_float,
                )

            set_status = assign_pixels(result_img, flat_pixels)
            try:
                result_img.use_view_as_render = False
            except Exception:
                pass
            if source_for_colorspace is not None:
                try:
                    result_img.colorspace_settings.name = (
                        source_for_colorspace.colorspace_settings.name
                    )
                except (AttributeError, TypeError):
                    pass
            try:
                result_img.alpha_mode = "STRAIGHT"
            except (AttributeError, TypeError):
                pass
            props.result_image = result_img

            rb = _readback_minmax(props.result_image)
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

            # Auto-save packed result to disk.
            stem = base_target_path.stem or "ChannelPacked"
            out_dir = (
                base_target_path.parent if str(base_target_path.parent) else Path(".")
            )
            if stem == "ChannelPacked" or stem.endswith("_packed"):
                out_stem = stem
            else:
                out_stem = f"{stem}_packed"
            out_path = _unique_path(out_dir / f"{out_stem}{ext}")

            props.result_image.filepath_raw = str(out_path)
            try:
                props.result_image.file_format = file_format
            except (AttributeError, TypeError):
                pass
            try:
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

        options_box = layout.box()
        options_box.label(text="Image Options", icon="PREFERENCES")
        row = options_box.row(align=True)
        row.prop(props, "auto_scale_to_largest", text="Auto Scale to Largest")
        override_row = row.row(align=True)
        override_row.enabled = props.mode == "SINGLE"
        override_row.prop(props, "override_rgb", text="Override RGB", toggle=True)
        sub = options_box.row(align=True)
        sub.enabled = props.auto_scale_to_largest
        sub.prop(props, "aspect_ratio_tolerance", text="Aspect Tolerance")

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
        result_box.operator("channelpacker.pack_channels", icon="FILE_BLEND")

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

        saved_row = result_box.row()
        saved_row.enabled = False
        saved_row.prop(props, "last_saved_path")

        if props.debug_info:
            debug_row = result_box.row()
            debug_row.enabled = False
            debug_row.prop(props, "debug_info", text="Debug")


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
classes = (
    BeyondChannelPackerProperties,
    ChannelPackerOTPackChannels,
    ChannelPackerOTLoadImage,
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
