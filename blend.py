import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

def extract_person(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Simulate segmentation: Let's assume green screen background
    lower = np.array([0, 200, 0])  # green threshold
    upper = np.array([150, 255, 150])
    
    mask = cv2.inRange(img_rgb, lower, upper)
    mask_inv = cv2.bitwise_not(mask)

    person_only = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_inv)
    
    return person_only, mask_inv


def estimate_light_direction(scene_path):
    img = cv2.imread(scene_path, 0)  # grayscale
    edges = cv2.Canny(img, 50, 150)
    
    # Use Hough Transform to detect lines (shadows)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    angles = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(np.degrees(angle))
    
    avg_angle = np.mean(angles) if angles else 0
    return avg_angle  # in degrees

def resize_person(person_img, mask, scale=0.5):
    h, w = person_img.shape[:2]
    resized = cv2.resize(person_img, (int(w*scale), int(h*scale)))
    mask = cv2.resize(mask, (int(w*scale), int(h*scale)))
    return resized, mask

def match_colors(person_img, background_img):
    matched = match_histograms(person_img, background_img, channel_axis=-1)

    return matched.astype(np.uint8)

def generate_shadow(mask, angle_deg, softness=21):
    # Shift the mask in the light direction
    dx = int(30 * np.cos(np.radians(angle_deg)))
    dy = int(30 * np.sin(np.radians(angle_deg)))

    # Translate the mask to cast shadow
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # Blur the shadow for softness
    shadow = cv2.GaussianBlur(shadow, (softness, softness), 0)

    # Convert to 3-channel and tint it dark
    shadow_colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        shadow_colored[:, :, i] = shadow // 3

    return shadow_colored


def feather_mask(mask, feather_amount=15):
    blur = cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0)
    normalized = blur / 255.0
    return normalized


import cv2
import numpy as np

def composite_person(bg_path, person_img, mask, shadow_img):
    """
    Places the person (with applied coloring and resizing) onto the background scene,
    applies shadow blending, and returns the final composite image.
    
    Parameters:
    - bg_path: Path to the background image
    - person_img: The color-matched person image (H, W, 3)
    - mask: The binary mask of the person (H, W)
    - shadow_img: The generated shadow image (H, W) or (H, W, 3)

    Returns:
    - final composite image
    """
    # Load the background image
    bg = cv2.imread(bg_path)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    bh, bw, _ = bg.shape
    ph, pw, _ = person_img.shape

    # Determine placement coordinates (centered)
    x = (bw - pw) // 2
    y = (bh - ph) // 2

    # Create a copy to preserve original
    composite = bg.copy()

    # --- SHADOW HANDLING ---

    # Extract background region for the shadow
    shadow_area = composite[y:y+ph, x:x+pw]

    # Resize shadow to match the person dimensions
    shadow_img = cv2.resize(shadow_img, (pw, ph))

    # If shadow is grayscale, convert to 3 channels
    if len(shadow_img.shape) == 2:
        shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_GRAY2RGB)
    elif shadow_img.shape[2] != 3:
        shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)

    # Ensure shadow_area and shadow_img match in shape
    if shadow_area.shape != shadow_img.shape:
        shadow_img = cv2.resize(shadow_img, (shadow_area.shape[1], shadow_area.shape[0]))

    # Blend shadow into background
    blended_shadow = cv2.addWeighted(shadow_area, 1.0, shadow_img, 0.4, 0)
    composite[y:y+ph, x:x+pw] = blended_shadow

    # --- PERSON COMPOSITING ---

    # Convert mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])

    # Place person image using the mask
    roi = composite[y:y+ph, x:x+pw]
    person_masked = cv2.bitwise_and(person_img, mask_3ch)
    bg_masked = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3ch))

    blended_person = cv2.add(bg_masked, person_masked)
    composite[y:y+ph, x:x+pw] = blended_person

    return composite

def remove_background_simple(image):
    """
    Very basic background removal using color thresholding.
    Assumes bright sky background (you may need to tune thresholds).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define lower and upper thresholds for bright backgrounds (like sky)
    lower = np.array([0, 0, 220])      # Light background
    upper = np.array([180, 60, 255])   # Adjust based on input image

    # Create mask: white where background, black where person
    bg_mask = cv2.inRange(hsv, lower, upper)

    # Invert mask: white where person
    person_mask = cv2.bitwise_not(bg_mask)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_DILATE, kernel)

    return person_mask

# Load and extract
person_img, person_mask = extract_person('person.jpg')
bg_img_path = 'scene.jpg'

# Estimate light, resize
angle = estimate_light_direction(bg_img_path)
resized_person, resized_mask = resize_person(person_img, person_mask, scale=0.5)

# Color match
colored_person = match_colors(resized_person, cv2.imread(bg_img_path))

# Shadow and compositing
shadow = generate_shadow(resized_mask, angle)
final = composite_person(bg_img_path, colored_person, resized_mask, shadow)

# Display
plt.imshow(final)
plt.axis('off')
plt.show()
