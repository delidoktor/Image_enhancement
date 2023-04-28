import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray

def adaptive_histogram_equalization(img, clip_limit=2.0, tile_grid_size=(4, 4)):
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
def unsharp_masking(img, ksize=(5, 5), sigma=1.0, amount=1.0):
    blurred_img = cv2.GaussianBlur(img, ksize, sigma)
    mask = cv2.subtract(img, blurred_img)
    return cv2.addWeighted(img, 0.15 + amount, mask, amount, -45)

def get_dark_channel(img, patch_size=15):
    min_channels = np.amin(img, axis=2)
    return cv2.erode(min_channels, np.ones((patch_size, patch_size)))

def estimate_atmospheric_light(img, dark_channel, top_percent=0.001):
    num_pixels = dark_channel.size
    num_brightest = int(num_pixels * top_percent)

    dark_channel_flat = dark_channel.flatten()
    brightest_indices = np.argpartition(dark_channel_flat, -num_brightest)[-num_brightest:]
    brightest_pixels = img.reshape((-1, 3))[brightest_indices]

    return np.amax(brightest_pixels, axis=0)

def compute_transmission_map(img, A, patch_size=15, omega=0.95):
    norm_img = img / A
    dark_channel_norm = get_dark_channel(norm_img, patch_size)
    return 1 - omega * dark_channel_norm

def soft_matting(img, transmission_map, radius=30, epsilon=1e-4):
    guided_filter = cv2.ximgproc.createGuidedFilter(img, radius, epsilon)
    refined_transmission = guided_filter.filter(transmission_map)
    return refined_transmission

def recover_image(img, A, transmission_map, t0=0.1):
    t = np.maximum(transmission_map, t0)[:, :, np.newaxis]
    J = (img - A) / t + A
    return np.clip(J, 0, 255).astype(np.uint8)

def remove_haze(img_path):
    img = io.imread(img_path)
    #histogram=adaptive_histogram_equalization(img) (used for combining both)
    #enhanced_img = unsharp_masking(histogram)      (used for combining both)
    img = img.astype(np.float32)
    dark_channel = get_dark_channel(img)
    A = estimate_atmospheric_light(img, dark_channel)
    transmission_map = compute_transmission_map(img, A)
    refined_transmission_map = soft_matting(img, transmission_map)
    dehazed_img = recover_image(img, A, refined_transmission_map)
    return dehazed_img

if __name__ == "__main__":
    input_image_path = "haze.png"
    output_image_path = "haze_result.png"
    dehazed_img = remove_haze(input_image_path)
    io.imsave(output_image_path, dehazed_img.astype(np.uint8))
    
