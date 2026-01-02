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

import bpy
import numpy as np

bl_info = {
    "name": "BeyondChannelPacker",
    "author": "Beyond Dev (Tyler Walker)",
    "version": (0, 1, 0),
    "blender": (2, 80, 0),
    "description": "Channel pack images in the Image Editor (choose RGB+Alpha or 4 separate channels).",
    "category": "Image",
}


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
            # Iterate over all screen areas to update the image editor display.
            for area in bpy.context.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    area.spaces.active.image = img
                    break

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
        description="If images differ in size but have similar aspect ratios, scale them to the largest image for packing",
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
        # Prepare an output array of shape (height, width, 4).
        # We always produce an RGBA output. If a channel image is missing,
        # default to 0 for colors and 1 for alpha.
        # -----------------------------------------------------------------------------
        output = np.zeros((height, width, 4), dtype=np.float32)

        # -----------------------------------------------------------------------------
        # Helper: Convert sRGB values to linear using a standard conversion formula.
        # -----------------------------------------------------------------------------
        def convert_srgb_to_linear(arr):
            return np.where(arr < 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

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
            tmp.scale(width, height)
            scaled_cache[img] = tmp
            temp_images.append(tmp)
            return tmp

        def get_pixels_array(img):
            img = get_image_for_target(img)
            if img is None:
                return None
            if not img.pixels:
                self.report({"ERROR"}, f"Image '{img.name}' has no pixel data.")
                return None
            arr = np.array(img.pixels[:], dtype=np.float32)
            denom = width * height
            if denom <= 0 or arr.size % denom != 0:
                self.report(
                    {"ERROR"},
                    f"Image '{img.name}' pixel buffer size doesn't match its dimensions.",
                )
                return None
            channels_in_buffer = arr.size // denom
            try:
                return arr.reshape((height, width, channels_in_buffer))
            except Exception as e:
                self.report({"ERROR"}, f"Error processing image '{img.name}': {e}")
                return None

        # -----------------------------------------------------------------------------
        # Helper: get a 2D array from an image by flattening its data to one channel.
        # A flag (is_alpha) prevents conversion on alpha channels.
        # -----------------------------------------------------------------------------
        def get_channel_data(img, is_alpha=False):
            if img is None:
                return None
            arr = get_pixels_array(img)
            if arr is None:
                return None
            # For multi‑channel images, take the first channel (assuming it is grayscale or R).
            data = arr[:, :, 0]
            # Convert color data from sRGB to linear if needed (do not convert alpha channels).
            if not is_alpha and img.colorspace_settings.name == "sRGB":
                data = convert_srgb_to_linear(data)
            return data

        # -----------------------------------------------------------------------------
        # Fill the output array based on the chosen mode.
        # -----------------------------------------------------------------------------
        try:
            if mode == "SINGLE":
                # Process the RGB image. It might have 3 or 4 channels.
                rgb_img = props.rgb_image
                arr_rgb = get_pixels_array(rgb_img)
                if arr_rgb is None:
                    return {"CANCELLED"}
                channels_rgb = arr_rgb.shape[2]
                # Use the first three channels. If only one channel is available, duplicate it.
                if channels_rgb >= 3:
                    if rgb_img.colorspace_settings.name == "sRGB":
                        output[:, :, 0] = convert_srgb_to_linear(arr_rgb[:, :, 0])
                        output[:, :, 1] = convert_srgb_to_linear(arr_rgb[:, :, 1])
                        output[:, :, 2] = convert_srgb_to_linear(arr_rgb[:, :, 2])
                    else:
                        output[:, :, 0] = arr_rgb[:, :, 0]
                        output[:, :, 1] = arr_rgb[:, :, 1]
                        output[:, :, 2] = arr_rgb[:, :, 2]
                else:
                    if rgb_img.colorspace_settings.name == "sRGB":
                        gray = convert_srgb_to_linear(arr_rgb[:, :, 0])
                    else:
                        gray = arr_rgb[:, :, 0]
                    output[:, :, 0] = gray
                    output[:, :, 1] = gray
                    output[:, :, 2] = gray

                # For alpha, use the provided alpha image if available; otherwise, default to opaque.
                if props.alpha_image is not None:
                    alpha_data = get_channel_data(props.alpha_image, is_alpha=True)
                    if alpha_data is None:
                        return {"CANCELLED"}
                    output[:, :, 3] = alpha_data
                else:
                    output[:, :, 3] = 1.0

            else:  # MULTI mode: use separate images for each channel.
                r_data = (
                    get_channel_data(props.r_image)
                    if props.r_image is not None
                    else None
                )
                g_data = (
                    get_channel_data(props.g_image)
                    if props.g_image is not None
                    else None
                )
                b_data = (
                    get_channel_data(props.b_image)
                    if props.b_image is not None
                    else None
                )
                a_data = (
                    get_channel_data(props.a_image, is_alpha=True)
                    if props.a_image is not None
                    else None
                )

                output[:, :, 0] = r_data if r_data is not None else 0.0
                output[:, :, 1] = g_data if g_data is not None else 0.0
                output[:, :, 2] = b_data if b_data is not None else 0.0
                output[:, :, 3] = a_data if a_data is not None else 1.0
        finally:
            for tmp in temp_images:
                if not tmp:
                    continue
                try:
                    bpy.data.images.remove(tmp, do_unlink=True)
                except (ReferenceError, RuntimeError):
                    pass

        # -----------------------------------------------------------------------------
        # Premultiply the RGB channels by the alpha channel to avoid white fringes in transparent areas.
        # -----------------------------------------------------------------------------
        output[:, :, :3] *= output[:, :, 3:4]

        # Flatten the output array to a 1D list (row‑major order) for Blender.
        flat_pixels = output.flatten()

        # -----------------------------------------------------------------------------
        # Create a new image in Blender to hold the packed data.
        # float_buffer=True creates a 32‑bit float image; export options will allow
        # you to save as 8‑bit or 16‑bit later.
        # -----------------------------------------------------------------------------
        result_img = bpy.data.images.new(
            name="ChannelPacked",
            width=width,
            height=height,
            alpha=True,
            float_buffer=True,
        )
        result_img.pixels = flat_pixels.tolist()

        # -----------------------------------------------------------------------------
        # If override is enabled in SINGLE mode, replace the original RGB image.
        # -----------------------------------------------------------------------------
        if mode == "SINGLE" and props.override_rgb:
            old_name = props.rgb_image.name
            bpy.data.images.remove(props.rgb_image, do_unlink=True)
            result_img.name = old_name
            props.rgb_image = result_img

        props.result_image = result_img
        for area in bpy.context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = result_img
                break

        self.report({"INFO"}, "Channel packing complete.")
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# Operator: Save Packed Image
# -----------------------------------------------------------------------------
class CHANNELPACKER_OT_save_image(bpy.types.Operator):
    """Save the packed image using Blender's save image functionality"""

    bl_idname = "channelpacker.save_image"
    bl_label = "Save Packed Image"

    def execute(self, context):
        props = context.scene.beyond_channel_packer
        if props.result_image is None:
            self.report({"ERROR"}, "No packed image to save.")
            return {"CANCELLED"}

        # Set the active image in the image editor to the result.
        for area in bpy.context.screen.areas:
            if area.type == "IMAGE_EDITOR":
                area.spaces.active.image = props.result_image
                break

        # Invoke Blender's built-in save-as operator.
        bpy.ops.image.save_as("INVOKE_DEFAULT")
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

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        allowed = {"rgb_image", "alpha_image", "r_image", "g_image", "b_image", "a_image"}
        if self.target_prop not in allowed:
            self.report({"ERROR"}, "Invalid target image slot.")
            return {"CANCELLED"}

        props = context.scene.beyond_channel_packer
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except Exception as e:
            self.report({"ERROR"}, f"Could not load image: {e}")
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
