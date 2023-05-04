input_image = imread('2.2.07.tiff');
input_image_double = im2double(input_image);
gamma = 3.33;
gamma_corrected_image = input_image_double .^ gamma;
brightening_factor = 1.5;
brightened_image = gamma_corrected_image * brightening_factor;
brightened_image(brightened_image > 1) = 1;
output_image = im2uint8(brightened_image);

figure;
subplot(1, 2, 1);
imshow(input_image);
title('Original Image');

subplot(1, 2, 2);
imshow(output_image);
title('Gamma Corrected and Brightened Image');

imwrite(output_image, 'output_image.jpg');
