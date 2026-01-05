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

bl_info = {
    "name": "BeyondChannelPacker",
    "author": "Beyond Dev (Tyler Walker)",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "description": "Channel pack images in the Image Editor (choose RGB+Alpha or 4 separate channels).",
    "category": "Image",
}


# -----------------------------------------------------------------------------
# Compatibility / helpers
# -----------------------------------------------------------------------------
def _get_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None
    return np


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
        description="Image providing the Alpha channel (will be flattened to one channel)",
        update=image_update_callback("alpha_image"),
    )
    # New property to override the original RGB image with the packed image.
    override_rgb: bpy.props.BoolProperty(
        name="Override RGB",
        description="Override the original RGB image with the packed image after packing",
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
        description="Allowed relative aspect ratio difference for auto scaling (e.g. 0.02 = 2%)",
        default=0.02,
        min=0.0,
        soft_max=0.1,
        max=0.5,
    )


# -----------------------------------------------------------------------------
# Operator: Pack Channels
# -----------------------------------------------------------------------------
class CHANNELPACKER_OT_pack_channels(bpy.types.Operator):
    """Pack selected channels into a new image"""

    bl_idname = "channelpacker.pack_channels"
    bl_label = "Pack Channels"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        mode = props.mode

        # -----------------------------------------------------------------------------
        # Determine target width/height.
        # If images differ in dimensions but have a similar aspect ratio, optionally
        # scale them to match the largest image for packing.
        # -----------------------------------------------------------------------------
        width = height = None

        def get_selected_images():
            if mode == "SINGLE":
                return [img for img in [props.rgb_image, props.alpha_image] if img]
            return [
                img for img in [props.r_image, props.g_image, props.b_image, props.a_image] if img
            ]

        selected_images = get_selected_images()

        # For SINGLE mode, the RGB image is required.
        if mode == "SINGLE":
            if props.rgb_image is None:
                self.report({"ERROR"}, "RGB Image is required for RGB + Alpha mode.")
                return {"CANCELLED"}
        else:
            if not selected_images:
                self.report(
                    {"ERROR"},
                    "At least one channel image must be provided in Separate Channels mode.",
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
            rel_diff = abs(ratio - target_ratio) / target_ratio if target_ratio else 0.0
            if rel_diff > tol:
                self.report({"ERROR"}, "All images must have the same dimensions.")
                return {"CANCELLED"}

        if width is None or height is None:
            self.report({"ERROR"}, "Could not determine image dimensions.")
            return {"CANCELLED"}

        # -----------------------------------------------------------------------------
        # Prepare an output buffer of RGBA floats (row-major order).
        # Defaults: RGB=0, A=1.
        # -----------------------------------------------------------------------------
        np = _get_numpy()
        pixel_count = width * height
        if pixel_count <= 0:
            self.report({"ERROR"}, "Could not determine image dimensions.")
            return {"CANCELLED"}

        if np is not None:
            output = np.zeros((height, width, 4), dtype=np.float32)
            output[:, :, 3] = 1.0
        else:
            output = [0.0] * (pixel_count * 4)
            for i in range(pixel_count):
                output[(i * 4) + 3] = 1.0

        scaled_cache = {}
        temp_images = []

        def get_image_for_target(img):
            if img is None:
                return None
            if img.size[0] == width and img.size[1] == height:
                return img
            cached = scaled_cache.get(img)
            if cached is not None:
                return cached
            tmp = img.copy()
            tmp.name = f"{img.name}_BCP_TMP"
            tmp.use_fake_user = False
            try:
                tmp.scale(width, height)
            except RuntimeError as exc:
                self.report({"ERROR"}, f"Could not scale image '{img.name}': {exc}")
                return None
            scaled_cache[img] = tmp
            temp_images.append(tmp)
            return tmp

        def get_pixels_flat(img):
            img = get_image_for_target(img)
            if img is None:
                return None, 0
            try:
                pixels = img.pixels[:]
            except RuntimeError as exc:
                self.report({"ERROR"}, f"Image '{img.name}' pixel data unavailable: {exc}")
                return None, 0

            denom = width * height
            if denom <= 0 or (len(pixels) % denom) != 0:
                self.report(
                    {"ERROR"},
                    f"Image '{img.name}' pixel buffer size doesn't match its dimensions.",
                )
                return None, 0
            channels_in_buffer = len(pixels) // denom
            return pixels, channels_in_buffer

        def get_pixels_array(img):
            pixels, channels_in_buffer = get_pixels_flat(img)
            if pixels is None:
                return None
            if np is None:
                return pixels, channels_in_buffer
            arr = np.asarray(pixels, dtype=np.float32)
            try:
                return arr.reshape((height, width, channels_in_buffer))
            except ValueError as exc:
                self.report({"ERROR"}, f"Error processing image '{img.name}': {exc}")
                return None

        # -----------------------------------------------------------------------------
        # Helper: get a 2D array from an image by flattening its data to one channel.
        # When is_alpha=True and the source has an alpha channel, prefer it.
        # -----------------------------------------------------------------------------
        def get_channel_data(img, is_alpha=False):
            if img is None:
                return None
            arr = get_pixels_array(img)
            if arr is None:
                return None
            if np is None:
                pixels, channels_in_buffer = arr
                channel_index = 3 if (is_alpha and channels_in_buffer >= 4) else 0
                return pixels, channels_in_buffer, channel_index

            channels_in_buffer = arr.shape[2]
            channel_index = 3 if (is_alpha and channels_in_buffer >= 4) else 0
            return arr[:, :, channel_index]

        # -----------------------------------------------------------------------------
        # Fill the output array based on the chosen mode.
        # -----------------------------------------------------------------------------
        try:
            if mode == "SINGLE":
                rgb_img = props.rgb_image
                arr_rgb = get_pixels_array(rgb_img)
                if arr_rgb is None:
                    return {"CANCELLED"}
                if np is not None:
                    channels_rgb = arr_rgb.shape[2]
                    if channels_rgb >= 3:
                        output[:, :, 0] = arr_rgb[:, :, 0]
                        output[:, :, 1] = arr_rgb[:, :, 1]
                        output[:, :, 2] = arr_rgb[:, :, 2]
                    else:
                        gray = arr_rgb[:, :, 0]
                        output[:, :, 0] = gray
                        output[:, :, 1] = gray
                        output[:, :, 2] = gray
                else:
                    rgb_pixels, channels_rgb = arr_rgb
                    for i in range(pixel_count):
                        out_base = i * 4
                        in_base = i * channels_rgb
                        if channels_rgb >= 3:
                            output[out_base + 0] = rgb_pixels[in_base + 0]
                            output[out_base + 1] = rgb_pixels[in_base + 1]
                            output[out_base + 2] = rgb_pixels[in_base + 2]
                        else:
                            gray = rgb_pixels[in_base + 0]
                            output[out_base + 0] = gray
                            output[out_base + 1] = gray
                            output[out_base + 2] = gray

                # For alpha, use the provided alpha image if available; otherwise, default to opaque.
                if props.alpha_image is not None:
                    alpha_data = get_channel_data(props.alpha_image, is_alpha=True)
                    if alpha_data is None:
                        return {"CANCELLED"}
                    if np is not None:
                        output[:, :, 3] = alpha_data
                    else:
                        alpha_pixels, channels_a, channel_index = alpha_data
                        for i in range(pixel_count):
                            output[(i * 4) + 3] = alpha_pixels[(i * channels_a) + channel_index]

            else:  # MULTI mode: use separate images for each channel.
                r_data = get_channel_data(props.r_image) if props.r_image is not None else None
                g_data = get_channel_data(props.g_image) if props.g_image is not None else None
                b_data = get_channel_data(props.b_image) if props.b_image is not None else None
                a_data = (
                    get_channel_data(props.a_image, is_alpha=True)
                    if props.a_image is not None
                    else None
                )

                if np is not None:
                    if r_data is not None:
                        output[:, :, 0] = r_data
                    if g_data is not None:
                        output[:, :, 1] = g_data
                    if b_data is not None:
                        output[:, :, 2] = b_data
                    if a_data is not None:
                        output[:, :, 3] = a_data
                else:
                    if r_data is not None:
                        r_pixels, channels_r, channel_index = r_data
                        for i in range(pixel_count):
                            output[(i * 4) + 0] = r_pixels[(i * channels_r) + channel_index]
                    if g_data is not None:
                        g_pixels, channels_g, channel_index = g_data
                        for i in range(pixel_count):
                            output[(i * 4) + 1] = g_pixels[(i * channels_g) + channel_index]
                    if b_data is not None:
                        b_pixels, channels_b, channel_index = b_data
                        for i in range(pixel_count):
                            output[(i * 4) + 2] = b_pixels[(i * channels_b) + channel_index]
                    if a_data is not None:
                        a_pixels, channels_a, channel_index = a_data
                        for i in range(pixel_count):
                            output[(i * 4) + 3] = a_pixels[(i * channels_a) + channel_index]
        finally:
            for tmp in temp_images:
                if not tmp:
                    continue
                try:
                    context.blend_data.images.remove(tmp, do_unlink=True)
                except (ReferenceError, RuntimeError):
                    pass

        if np is not None:
            flat_pixels = output.reshape(-1).tolist()
        else:
            flat_pixels = output

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
            rgb_img.pixels = flat_pixels
            props.result_image = rgb_img

            # If the RGB image is file-backed and is a PNG, overwrite it on disk.
            filepath = getattr(rgb_img, "filepath_raw", "") or ""
            if filepath.lower().endswith(".png"):
                try:
                    rgb_img.file_format = "PNG"
                except (AttributeError, TypeError):
                    pass
                try:
                    rgb_img.save()
                except RuntimeError as exc:
                    self.report({"WARNING"}, f"Packed, but could not overwrite PNG: {exc}")
            else:
                self.report(
                    {"WARNING"},
                    "Packed into the RGB image datablock, but did not auto-save (source is not a .png).",
                )
        else:
            # -----------------------------------------------------------------------------
            # Create a new image datablock for the packed result.
            # -----------------------------------------------------------------------------
            result_img = context.blend_data.images.new(
                name="ChannelPacked",
                width=width,
                height=height,
                alpha=True,
                float_buffer=True,
            )
            result_img.pixels = flat_pixels
            if mode == "SINGLE" and props.rgb_image is not None:
                try:
                    result_img.colorspace_settings.name = props.rgb_image.colorspace_settings.name
                except (AttributeError, TypeError):
                    pass
                try:
                    result_img.alpha_mode = props.rgb_image.alpha_mode
                except (AttributeError, TypeError):
                    pass
            props.result_image = result_img

        _set_active_image_in_image_editor(context, props.result_image)

        self.report({"INFO"}, "Channel packing complete.")
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Operator: Save Packed Image
# -----------------------------------------------------------------------------
class CHANNELPACKER_OT_save_image(bpy.types.Operator):
    """Save the packed image to disk"""

    bl_idname = "channelpacker.save_image"
    bl_label = "Save Packed Image"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.bmp;*.exr;*.hdr",
        options={"HIDDEN"},
    )

    def invoke(self, context, _event):
        props = context.scene.beyond_channel_packer
        img = props.result_image
        if img is None:
            self.report({"ERROR"}, "No packed image to save.")
            return {"CANCELLED"}

        filepath = (getattr(img, "filepath_raw", "") or getattr(img, "filepath", "") or "").strip()
        if not filepath:
            filepath = "channel_packed.png"
        if not Path(filepath).suffix:
            filepath = f"{filepath}.png"
        self.filepath = filepath

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        img = props.result_image
        if img is None:
            self.report({"ERROR"}, "No packed image to save.")
            return {"CANCELLED"}

        filepath = (self.filepath or "").strip()
        if not filepath:
            self.report({"ERROR"}, "No filepath selected.")
            return {"CANCELLED"}

        file_format = _image_file_format_from_path(filepath)
        if file_format is None:
            self.report({"ERROR"}, "Unsupported file extension for saving.")
            return {"CANCELLED"}

        _set_active_image_in_image_editor(context, img)

        img.filepath_raw = filepath
        try:
            img.file_format = file_format
        except (AttributeError, TypeError):
            pass

        try:
            img.save()
        except RuntimeError as exc:
            self.report({"ERROR"}, f"Could not save image: {exc}")
            return {"CANCELLED"}

        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Operator: Load Image From Disk (File Browser)
# -----------------------------------------------------------------------------
class CHANNELPACKER_OT_load_image(bpy.types.Operator):
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
        allowed = {"rgb_image", "alpha_image", "r_image", "g_image", "b_image", "a_image"}
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
class CHANNELPACKER_PT_panel(bpy.types.Panel):
    """Panel for channel packing in the Image Editor"""

    bl_label = "Beyond Channel Packer"
    bl_idname = "CHANNELPACKER_PT_panel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Channel Packer"

    def draw(self, context):
        layout = self.layout
        props = context.scene.beyond_channel_packer

        # Mode selection (RGB+Alpha vs. Separate Channels)
        layout.prop(props, "mode")
        row = layout.row(align=True)
        row.prop(props, "auto_scale_to_largest")
        sub = row.row(align=True)
        sub.enabled = props.auto_scale_to_largest
        sub.prop(props, "aspect_ratio_tolerance", text="Tol")

        # Display the appropriate settings based on the selected mode.
        if props.mode == "SINGLE":
            box = layout.box()
            box.label(text="RGB + Alpha Mode", icon="IMAGE_DATA")
            row = box.row(align=True)
            row.prop(props, "rgb_image")
            op = row.operator("channelpacker.load_image", text="", icon="FILE_FOLDER")
            op.target_prop = "rgb_image"
            if props.rgb_image:
                box.template_preview(props.rgb_image, show_buttons=False)
            row = box.row(align=True)
            row.prop(props, "alpha_image")
            op = row.operator("channelpacker.load_image", text="", icon="FILE_FOLDER")
            op.target_prop = "alpha_image"
            if props.alpha_image:
                box.template_preview(props.alpha_image, show_buttons=False)
            row = box.row()
            row.prop(props, "override_rgb")
        else:
            box = layout.box()
            box.label(text="Separate Channels Mode", icon="IMAGE_DATA")
            row = box.row(align=True)
            row.prop(props, "r_image")
            op = row.operator("channelpacker.load_image", text="", icon="FILE_FOLDER")
            op.target_prop = "r_image"
            if props.r_image:
                box.template_preview(props.r_image, show_buttons=False)
            row = box.row(align=True)
            row.prop(props, "g_image")
            op = row.operator("channelpacker.load_image", text="", icon="FILE_FOLDER")
            op.target_prop = "g_image"
            if props.g_image:
                box.template_preview(props.g_image, show_buttons=False)
            row = box.row(align=True)
            row.prop(props, "b_image")
            op = row.operator("channelpacker.load_image", text="", icon="FILE_FOLDER")
            op.target_prop = "b_image"
            if props.b_image:
                box.template_preview(props.b_image, show_buttons=False)
            row = box.row(align=True)
            row.prop(props, "a_image")
            op = row.operator("channelpacker.load_image", text="", icon="FILE_FOLDER")
            op.target_prop = "a_image"
            if props.a_image:
                box.template_preview(props.a_image, show_buttons=False)

        layout.separator()
        # Button to pack channels.
        layout.operator("channelpacker.pack_channels", icon="FILE_BLEND")

        # If a packed image exists, show a preview.
        if props.result_image:
            layout.label(text="Result Preview:")
            layout.template_preview(props.result_image, show_buttons=False)

        layout.separator()
        # Button to save the packed image.
        layout.operator("channelpacker.save_image", icon="IMAGE")


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
classes = (
    BeyondChannelPackerProperties,
    CHANNELPACKER_OT_pack_channels,
    CHANNELPACKER_OT_save_image,
    CHANNELPACKER_OT_load_image,
    CHANNELPACKER_PT_panel,
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
