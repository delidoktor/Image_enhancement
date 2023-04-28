import cv2
import numpy as np
# not used because useless for dehazing
def remove_salt_noise(img, ksize=3):
    return cv2.medianBlur(img, ksize)
#used
def unsharp_masking(img, ksize=(5, 5), sigma=1.0, amount=1.0):
    blurred_img = cv2.GaussianBlur(img, ksize, sigma)
    mask = cv2.subtract(img, blurred_img)
    return cv2.addWeighted(img, 0.15 + amount, mask, amount, -45)
#used
def adaptive_histogram_equalization(img, clip_limit=2.0, tile_grid_size=(4, 4)):
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
# not used because useless for dehazing
def high_boost_filter(img, k=1.0):
    # Apply Gaussian blur to the input image
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Compute the mask by subtracting the blurred image from the original image
    mask = cv2.subtract(img, blurred_img)
    
    # Add the mask to the original image, scaled by the boost factor k
    return cv2.addWeighted(img, 1 + k, mask, k, 0)

def enhance_image(input_image_path, output_image_path):
    # Read the input image
    img = cv2.imread(input_image_path)

    # Apply adaptive histogram equalization (CLAHE)
    equalized_img = adaptive_histogram_equalization(img)

    # Apply high boost filter for image sharpening
    enhanced_img = unsharp_masking(equalized_img)

    # Save the enhanced image
    cv2.imwrite(output_image_path,enhanced_img)

if __name__ == "__main__":
    input_image_path = "haze.png"  
    output_image_path = "haze_result.png"  
    enhance_image(input_image_path, output_image_path)
