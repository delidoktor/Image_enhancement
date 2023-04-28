import cv2
import numpy as np

def count_pools(input_image_path):
    # read image
    image = cv2.imread(input_image_path)

    # to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #range of blue color in HSV
    lower_blue = np.array([85, 60, 60])
    upper_blue = np.array([130, 255, 255])

    # range of dark blue color in HSV
    lower_dark_blue = np.array([85, 100, 0])
    upper_dark_blue = np.array([130, 255, 200])

    # range of white color in HSV
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([255, 30, 255])

    # threshold the HSV image to get only desired color space
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    dark_blue_mask = cv2.inRange(hsv_image, lower_dark_blue, upper_dark_blue)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # combine masks
    combined_mask = cv2.bitwise_or(blue_mask, white_mask)

    # gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # draw contours in the image
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter small contours by setting a threshold
    min_area = 130
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    # classify and count contours
    total_count = len(filtered_contours)
    shaded_count = 0
    closed_count = 0
    pool_types = []

    for contour in filtered_contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if cv2.pointPolygonTest(contour, (cX, cY), False) >= 0:
            if dark_blue_mask[cY, cX] > 0:
                shaded_count += 1
                pool_types.append("shaded")
            elif white_mask[cY, cX] > 0:
                closed_count += 1
                pool_types.append("closed")
            else:
                pool_types.append("normal")

    # draw filtered contours 
    contour_image = image.copy()
    for i, contour in enumerate(filtered_contours):
        if pool_types[i] == "shaded":
            contour_color = (255, 0, 0) # Red for shaded pools
        elif pool_types[i] == "closed":
            contour_color = (0, 255, 0) # Green for closed pools
        else:
            contour_color = (0, 0, 255) # Blue for normal pools

        cv2.drawContours(contour_image, [contour], -1, contour_color, 2)

    height, width, _ = contour_image.shape
    text_position = (width - 700, height - 20)
    total_text = f'Total count: {total_count}'
    normal_text = f' | Normal: {total_count - shaded_count - closed_count}'
    shaded_text = f' | Shaded: {shaded_count}'
    closed_text = f' | Closed: {closed_count}'
    
    # set colors for  text
    total_color = (255, 0, 255)  # Purple for total count
    normal_color = (0, 0, 255)  # Blue for normal pools
    shaded_color = (255, 0, 0)  # Red for shaded pools
    closed_color = (0, 255, 0)  # Green for closed pools
    
    # write text with the colors
    cv2.putText(contour_image, total_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, total_color, 2)
    cv2.putText(contour_image, normal_text, (text_position[0] + 160, text_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, normal_color, 2)
    cv2.putText(contour_image, shaded_text, (text_position[0] + 320, text_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shaded_color, 2)
    cv2.putText(contour_image, closed_text, (text_position[0] + 480, text_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, closed_color, 2)
    
    # display
    cv2.imshow('Contour Image', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_image_path,contour_image)

    return total_count, shaded_count, closed_count


if __name__ == '__main__':
    input_image_path = 'moliets.png'
    output_image_path = "moliets_result.png"
    num_pools,shaded,closed = count_pools(input_image_path)
    print(f'Number of pools detected: {num_pools}')
    print(f'Number of normal pools : {num_pools-(shaded+closed)}')
    print(f'Number of shaded pools : {shaded}')
    print(f'Number of closed pools : {closed}')

