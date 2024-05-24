<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Image Compression using Huffman Codebook</h1>
    <p>Welcome to the repository for our project on image compression using the Huffman codebook. This project involves compressing images by first dividing them into blocks, quantizing the images, and then using Huffman coding based on the probabilities of the quantized values.</p>
    <h2>Project Steps</h2>
    <ol>
        <li><strong>Image Division:</strong> The input image is divided into smaller blocks.</li>
        <li><strong>Cropping:</strong> A small portion of the image is cropped for further processing.</li>
        <li><strong>Quantization:</strong> The cropped image is quantized into different levels to reduce the number of unique values.</li>
        <li><strong>Huffman Codebook Construction:</strong> The Huffman code is built according to the probabilities of the quantized values.</li>
        <li><strong>Compression:</strong> The image is compressed using the Huffman code generated in the previous step.</li>
        <li><strong>Decompression:</strong> The compressed file is used to reconstruct the image using the Huffman code.</li>
    </ol>
    <h2>Directory Structure</h2>
    <h2>Installation</h2>
    <p>To use this project, clone the repository and install the required dependencies:</p>
    <h2>Usage</h2>
    <p>Follow the steps below to compress and decompress an image:</p>
    <ol>
        <li>Place the image you want to compress </li>
        <li>Run the code1 <code>1.build_huffman_codebook.py</code> to build the Huffman codebook according to your image and decompress the image using your Huffman codebook:</li>
        <li>Run the code2<code>2.huffman_inbuilt_codebook.py</code> to use the inbuilt library of Python for building Huffman codebook:</li>
        <li>Run the <code>3.decompress_huffman_inbuilt_text_files.py</code> to decompress the images compressed by code2</li>
    </ol>
    <h2>Contributing</h2>
    <p>I welcome contributions! Please fork the repository and submit a pull request with your changes. Make sure to add tests for any new features or bug fixes.</p>
    <h2>License</h2>
    <p>This project is licensed under the MIT License. See the LICENSE file for more details.</p>
</body>
</html>
