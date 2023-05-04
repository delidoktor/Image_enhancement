% Read input image
inputImageFile = '2.2.07.tiff'; % Provide the path to your input image
inputImage = imread(inputImageFile);

% Convert the input image to double format for calculations
inputImageDouble = im2double(inputImage);

% Set gamma value
gamma = 3.33; % You can modify this value as desired

% Apply gamma correction
outputImageDouble = inputImageDouble .^ gamma;

% Convert the corrected image back to uint8 format
outputImage = im2uint8(outputImageDouble);

% Display input and output images
figure;
subplot(1, 2, 1);
imshow(inputImage);
title('Input Image');

subplot(1, 2, 2);
imshow(outputImage);
title('Gamma Corrected Image');

% Save output image
outputImageFile = 'gamma_corrected_image.jpg';
imwrite(outputImage, outputImageFile);
