import streamlit as st
import numpy as np
import cv2
from rembg import remove
from PIL import Image
import tempfile
import io

# Page config
st.set_page_config(layout="wide")
st.title("🖼️ Person Background Removal & Smart Composite with Shadows")

# Upload images
person_file = st.file_uploader("Upload Person Image", type=["png", "jpg", "jpeg"])
bg_file = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])

if person_file and bg_file:
    # Load and remove background
    person_img = Image.open(person_file).convert("RGBA")
    person_np = np.array(person_img)
    no_bg_img = remove(person_img)
    no_bg_np = np.array(no_bg_img)

    # Save no-background version temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        Image.fromarray(no_bg_np).save(tmp.name)
        tmp_path = tmp.name

    # Shadow detection
    person_bgr = cv2.cvtColor(np.array(person_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2GRAY)
    _, shadow_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(gray, 30, 100)
    shadow_edges = cv2.bitwise_and(edges, shadow_mask)
    edge_density = np.count_nonzero(shadow_edges) / (np.count_nonzero(shadow_mask) + 1e-5)
    classification = "Hard Shadow" if edge_density > 0.05 else "Soft Shadow"

    hard_shadow_mask = shadow_mask if classification == "Hard Shadow" else np.zeros_like(shadow_mask)
    soft_shadow_mask = shadow_mask if classification == "Soft Shadow" else np.zeros_like(shadow_mask)

    # Load background and prepare composition
    bg_img = Image.open(bg_file).convert("RGB")
    bg_np = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGR)

    # Load person with alpha
    person_rgba = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(person_rgba)
    person_rgb = cv2.merge((b, g, r))
    mask = a

    # Resize person to fit on background
    x_offset, bottom_margin = 800, 50
    max_h = bg_np.shape[0] - bottom_margin
    max_w = bg_np.shape[1] - x_offset
    scale = min(0.9, max_h / person_rgb.shape[0], max_w / person_rgb.shape[1])
    new_size = (int(person_rgb.shape[1] * scale), int(person_rgb.shape[0] * scale))

    person_rgb = cv2.resize(person_rgb, new_size)
    mask = cv2.resize(mask, new_size)
    y_offset = bg_np.shape[0] - person_rgb.shape[0] - bottom_margin

    # Blend person onto background
    roi = bg_np[y_offset:y_offset + person_rgb.shape[0], x_offset:x_offset + person_rgb.shape[1]]
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    blended = (roi.astype(np.float32) * (1 - mask_3ch) + person_rgb.astype(np.float32) * mask_3ch).astype(np.uint8)

    final = bg_np.copy()
    final[y_offset:y_offset + blended.shape[0], x_offset:x_offset + blended.shape[1]] = blended

    # Add soft shadow
    shadow = cv2.GaussianBlur(mask, (45, 45), sigmaX=30)
    shadow = (shadow / 255.0 * 90).astype(np.uint8)
    shadow_layer = np.zeros_like(bg_np)
    shadow_color = cv2.merge([shadow, shadow, shadow])

    sh_x = x_offset + 30
    sh_y = y_offset + 20
    sh_h, sh_w = min(shadow.shape[0], bg_np.shape[0] - sh_y), min(shadow.shape[1], bg_np.shape[1] - sh_x)
    shadow_color = shadow_color[:sh_h, :sh_w]
    shadow_region = shadow_layer[sh_y:sh_y + sh_h, sh_x:sh_x + sh_w]

    np.copyto(shadow_region, shadow_color, where=shadow_color > 0)
    final = cv2.subtract(final, shadow_layer)

    # LAB Histogram Similarity
    person_lab = cv2.cvtColor(person_rgb, cv2.COLOR_BGR2LAB)
    bg_resized = cv2.resize(bg_np, (person_rgb.shape[1], person_rgb.shape[0]))
    bg_lab = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2LAB)

    similarity = [
        cv2.compareHist(cv2.calcHist([person_lab], [i], None, [256], [0, 256]),
                        cv2.calcHist([bg_lab], [i], None, [256], [0, 256]),
                        cv2.HISTCMP_CORREL)
        for i in range(3)
    ]

    # ---------- DISPLAY ----------
    st.subheader("📊 LAB Histogram Similarity")
    st.write(f"**L:** {similarity[0]:.4f}, **A:** {similarity[1]:.4f}, **B:** {similarity[2]:.4f}")
    st.write(f"🌓 Shadow Classification: **{classification}** (Edge density: {edge_density:.4f})")

    col1, col2 = st.columns(2)
    col1.image(person_img, caption="Original Person Image", use_column_width=True)
    col2.image(no_bg_np, caption="Background Removed", use_column_width=True)

    col3, col4 = st.columns(2)
    col3.image(hard_shadow_mask, caption="Hard Shadow Mask", use_column_width=True, clamp=True)
    col4.image(soft_shadow_mask, caption="Soft Shadow Mask", use_column_width=True, clamp=True)

    st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Final Composite with Shadow", use_column_width=True)

    # Download final image
    final_pil = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    final_pil.save(buffer, format="PNG")
    st.download_button("⬇️ Download Final Image", data=buffer.getvalue(), file_name="final_composite.png", mime="image/png")
