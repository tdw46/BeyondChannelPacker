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
        # Determine a reference image to get width and height.
        # All provided images (if any) must share these dimensions.
        # -----------------------------------------------------------------------------
        width = height = None

        def update_dimensions(img):
            nonlocal width, height
            if img is not None:
                if width is None and height is None:
                    width, height = img.size[0], img.size[1]
                else:
                    if img.size[0] != width or img.size[1] != height:
                        self.report(
                            {"ERROR"}, "All images must have the same dimensions."
                        )
                        return False
            return True

        # For SINGLE mode, the RGB image is required.
        if mode == "SINGLE":
            if props.rgb_image is None:
                self.report({"ERROR"}, "RGB Image is required for RGB + Alpha mode.")
                return {"CANCELLED"}
            if not update_dimensions(props.rgb_image):
                return {"CANCELLED"}
            if props.alpha_image is not None:
                if not update_dimensions(props.alpha_image):
                    return {"CANCELLED"}
        else:  # MULTI mode: try to update dimensions from any provided channel.
            provided = False
            for img in [props.r_image, props.g_image, props.b_image, props.a_image]:
                if img is not None:
                    provided = True
                    if not update_dimensions(img):
                        return {"CANCELLED"}
            if not provided:
                self.report(
                    {"ERROR"},
                    "At least one channel image must be provided in Separate Channels mode.",
                )
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

        # -----------------------------------------------------------------------------
        # Helper: get a 2D array from an image by flattening its data to one channel.
        # A flag (is_alpha) prevents conversion on alpha channels.
        # -----------------------------------------------------------------------------
        def get_channel_data(img, is_alpha=False):
            if img is None:
                return None
            # Ensure the image has valid pixel data.
            if not img.pixels:
                self.report({"ERROR"}, f"Image '{img.name}' has no pixel data.")
                return None
            channels = img.channels
            arr = np.array(img.pixels[:], dtype=np.float32)
            try:
                arr = arr.reshape((height, width, channels))
            except Exception as e:
                self.report({"ERROR"}, f"Error processing image '{img.name}': {e}")
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
        if mode == "SINGLE":
            # Process the RGB image. It might have 3 or 4 channels.
            rgb_img = props.rgb_image
            channels_rgb = rgb_img.channels
            arr_rgb = np.array(rgb_img.pixels[:], dtype=np.float32)
            try:
                arr_rgb = arr_rgb.reshape((height, width, channels_rgb))
            except Exception as e:
                self.report({"ERROR"}, f"Error processing RGB image: {e}")
                return {"CANCELLED"}
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
                get_channel_data(props.r_image) if props.r_image is not None else None
            )
            g_data = (
                get_channel_data(props.g_image) if props.g_image is not None else None
            )
            b_data = (
                get_channel_data(props.b_image) if props.b_image is not None else None
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

        # Store the result and update the image editor view.
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

        # Display the appropriate settings based on the selected mode.
        if props.mode == "SINGLE":
            box = layout.box()
            box.label(text="RGB + Alpha Mode", icon="IMAGE_DATA")
            row = box.row()
            row.prop(props, "rgb_image")
            if props.rgb_image:
                box.template_preview(props.rgb_image, show_buttons=False)
            row = box.row()
            row.prop(props, "alpha_image")
            if props.alpha_image:
                box.template_preview(props.alpha_image, show_buttons=False)
        else:
            box = layout.box()
            box.label(text="Separate Channels Mode", icon="IMAGE_DATA")
            row = box.row()
            row.prop(props, "r_image")
            if props.r_image:
                box.template_preview(props.r_image, show_buttons=False)
            row = box.row()
            row.prop(props, "g_image")
            if props.g_image:
                box.template_preview(props.g_image, show_buttons=False)
            row = box.row()
            row.prop(props, "b_image")
            if props.b_image:
                box.template_preview(props.b_image, show_buttons=False)
            row = box.row()
            row.prop(props, "a_image")
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
