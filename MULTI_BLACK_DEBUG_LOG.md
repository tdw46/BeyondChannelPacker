# MULTI Mode “Black + Opaque” Debug Log

This file tracks what we’ve tried so far to fix the issue where **Separate
Channels (MULTI) mode** outputs a **completely black** image with **no
transparency**, while **RGB + Alpha (SINGLE) mode** works.

## Symptoms

- MULTI output: black RGB and opaque alpha in Blender (preview and saved file).
- SINGLE output: works as expected (correct RGB and alpha, scaling works).
- Inputs (MULTI):
  - `r_image/g_image/b_image`: greyscale PNGs loaded as RGBA (RGB identical),
    alpha appears white.
  - `a_image`: mask provided separately (greyscale).
  - Color space reported: sRGB.
  - Bit depth corrected: 8-bit PNGs (not 16-bit).

## UI / Workflow Changes (Not Root Cause)

- UI reorganized into boxed sections (Mode, Image Options, Source Images,
  Result), mode shown as “tab-like” buttons.
- Removed “Save Packed Image” operator/button and auto-saved the result
  instead; result preview shown as thumbnail.

## Packing Attempts Tried (and Failed for MULTI)

1. **Direct numpy reshape from `img.pixels[:]`**
   - Read `np.array(img.pixels[:], dtype=np.float32)`, reshape to
     `(height, width, channels)`, then fill output RGBA.
   - Result: MULTI still black/opaque.

2. **`pixels.foreach_get` for more reliable reads**
   - Switched to `img.pixels.foreach_get(arr)` to populate a float array.
   - Result: MULTI still black/opaque.

3. **Blender-side scaling via `Image.copy()` + `Image.scale()`**
   - Created temp images: `tmp = img.copy(); tmp.scale(width, height)` and read
     temp pixels.
   - Result: MULTI still black/opaque.
   - Also tested caching temps and cleaning them up.

4. **Channel extraction heuristics**
   - Tried extracting:
     - `arr[:, :, 0]` (red)
     - luminance-ish / max(R, G, B)
     - “prefer alpha if alpha varies and RGB is flat” heuristic
   - Result: MULTI still black/opaque.

5. **Ensure output has alpha**
   - Created result with `alpha=True`, forced `alpha_mode="STRAIGHT"`.
   - Result: MULTI still black/opaque.

6. **Write pixels with `foreach_set` + `image.update()`**
   - Switched result assignment to use `pixels.foreach_set(flat_pixels)` and
     call `image.update()` to force refresh.
   - Result: MULTI still black/opaque.

7. **Avoid Blender scaling; numpy resample instead (current approach)**
   - Removed temp `copy()/scale()` path.
   - Read pixels at source size, then nearest-neighbor resample in numpy to
     target `width/height`.
   - Result: MULTI still black/opaque.

8. **(Tried, then reverted) Compositor-based pack**
   - Implemented a compositor-node pipeline (`Separate RGBA` → `Combine RGBA`),
     saved to disk, then reloaded for preview.
   - Reverted: user requirement is “pure Python only; no compositor.”

## Current Approach (Pure Python)

- Uses `img.pixels.foreach_get()` (fallback to `img.pixels[:]`) to read a 1D
  float array, reshapes to `(h, w, channels)`, pads to RGBA, and resamples if
  needed.
- Writes pixels using `pixels.foreach_set()` (fallback to `pixels[:] = ...`) and
  calls both `image.update()` and `image.update_tag()`.
- If MULTI output RGB is still all-zero, prints per-input min/max stats to the
  Blender console to tell whether the failure is on read or write.

## New Attempt (Pure Python, Jan 2026)

Goal: avoid color-management/premultiply pitfalls when reading greyscale
channel maps that are tagged as sRGB.

- During MULTI reads only, temporarily forces:
  - `image.colorspace_settings.name = "Non-Color"`
  - `image.alpha_mode = "STRAIGHT"`
  - then restores the original settings immediately after reading.
- Output pixels are written as a plain Python list (old-code style), not a numpy
  array, to eliminate any possibility of numpy/foreach buffer issues.
- MULTI R/G/B extraction now also allows falling back to the image’s alpha
  channel when RGB is flat/zero (covers cases where greyscale ended up in alpha
  or RGB is effectively premultiplied away).

- Added a post-write readback check on the result image (`WRITE[...]`) to
  confirm whether Blender actually stored the pixels we computed (`OUT[...]`).
- Added a `SET[...]` status to show whether the pixel write succeeded via
  `foreach_set` or slice assignment, or failed with a specific error.
- Added `BUF[...]` diagnostics to confirm the result image pixel buffer length
  and whether it’s float-backed.

## What MULTI Is *Supposed* To Do

- Treat each input as greyscale and use it as a single channel:
  - R comes from `r_image`, G from `g_image`, B from `b_image`
  - A comes from `a_image` (mask)
- Then run the same “RGB + Alpha” output behavior (RGBA image, alpha preserved,
  autosave).

## Leading Hypotheses

These are consistent with “SINGLE works” but “MULTI reads as black”:

1. **Input pixel reads return zeros for the MULTI images only**
   - Could be that these specific images are not actually loaded into CPU memory
     when accessed via the add-on, even if visible in the UI.

2. **Input images have data in a different channel than expected**
   - Example: greyscale data might be in alpha, or RGB may be premultiplied to
     zero because alpha is zero (despite appearing white in the UI).

3. **Name collision / datablock reuse is showing an old “ChannelPacked”**
   - Blender may create `ChannelPacked.001` etc; if the panel is previewing a
     stale datablock, you’d keep seeing the old black result.

4. **Color management mismatch is hiding the signal**
   - Less likely to produce “completely black”, but possible if there’s a
     conversion step zeroing values.

5. **Newly-created images reject pixel writes**
   - Observed: computed output (`OUT`) is correct, but post-write readback
     (`WRITE`) stays at the default (0,0,0,1).
   - Next mitigation: avoid `bpy.data.images.new(...)` and instead write into a
     copy of an existing image datablock (mirrors SINGLE override behavior).

6. **Copied images may be RGB-only (alpha forced to 1)**
   - Observed: RGB writes succeed but alpha reads back as 1.0 everywhere.
   - Mitigation: base MULTI result on a known RGBA image datablock generated by
     writing a tiny RGBA PNG to `bpy.app.tempdir` at the target dimensions, then
     load+copy+write.

7. **New mitigation: generated 1×1 RGBA base**
   - Create a 1×1 generated image with `alpha=True`, then `scale(width,height)`
     to allocate an RGBA buffer, then write pixels into it.
   - Goal: avoid “RGB-only copies” while also avoiding file-loading buffers.

8. **New mitigation: loaded 1×1 RGBA base (then copy+scale)**
   - Load a tiny RGBA PNG (`BCP_RGBA_BASE`), force-load its pixel buffer, then
     copy it and scale the copy to target size before writing pixels.
   - Goal: result datablock supports alpha and also accepts pixel writes (copy
     path).

9. **New mitigation: generated RGBA then copy (no scaling)**
   - Create a generated RGBA image at the target size, immediately copy it,
     delete the temp, then write pixels into the copy.
   - Goal: keep alpha support while using the “copy” path that actually accepts
     pixel writes in this Blender build, without triggering file-buffer load
     errors during `scale()`.

10. **Pure-Python PNG output (bypass Blender pixel writes)**
   - Write the computed packed RGBA directly to a PNG on disk via a minimal PNG
     encoder (zlib + scanlines), then load that PNG for preview.
   - Goal: avoid Blender 5.0 image datablock pixel-write/alpha quirks entirely.

## Next Steps Planned

1. Confirm whether MULTI input reads return non-zero values (via console
   min/max output).
2. If reads are non-zero but result is black, isolate whether the issue is
   pixel assignment or image display settings by writing a known test pattern.

## Jan 2026 Follow-Up: File-Write Verification + sRGB Encode

To eliminate Blender datablock write quirks (where `foreach_set` reported OK but
readback stayed at defaults), MULTI now:

- Writes the packed result to disk via a tiny pure-Python PNG encoder
  (`_write_rgba_png8`) and loads that PNG for preview.
- Adds additional diagnostics:
  - `U8[...]` min/max of the 8-bit bytes that were written to the PNG.
  - `FILE[...]` min/max read back from Blender after loading the written PNG.
- Encodes RGB as sRGB when writing the PNG (linear → sRGB transfer), leaving
  alpha unmodified (straight alpha). This aims to match SINGLE mode’s “looks
  correct when viewed as sRGB” behavior.
