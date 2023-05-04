% Read the input image
input_image = imread('2.2.07.tiff');

% Convert the image to grayscale
grayscale_image = rgb2gray(input_image);

% Convert the grayscale image to double for further processing
grayscale_image_double = im2double(grayscale_image);

% Apply gamma correction
gamma = 3.33;
gamma_corrected_image = grayscale_image_double .^ gamma;

% Apply brightening adjustment
brightening_factor = 1.5; % You can adjust this value as needed
brightened_image = gamma_corrected_image * brightening_factor;

% Clip pixel values greater than 1 to 1
brightened_image(brightened_image > 1) = 1;

% Convert the grayscale image back to uint8 format
output_grayscale_image = im2uint8(brightened_image);

% Convert the grayscale image back to RGB format
output_image_2 = repmat(output_grayscale_image, [1, 1, 3]);

% Display the original and processed images
figure;
subplot(1, 2, 1);
imshow(input_image);
title('Original Image');

subplot(1, 2, 2);
imshow(output_image_2);
title('Grayscale, Gamma Corrected, and Brightened Image in RGB');

% Save the output image
imwrite(output_image_2, 'output_image_2.jpg');
