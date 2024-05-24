import numpy as np
import cv2 as cv
from collections import defaultdict
import huffman

'''
From this code,
 Get the huffman codebook using inbuilt huffman libarary
 Encoded the image and save into text files
 Calculate average length , compression ratios using inbuilt huffman algorithm
'''

image = cv.imread('Sample1.jpg')   # Read the image

class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq
def quantize_matrix(matrix, quantization_levels):
    # Define the quantization range
    min_range, max_range = 0, 256

    # Calculate the width of each quantization level
    level_width = (max_range - min_range) / len(quantization_levels)

    # Quantize the matrix
    quantized_matrix = np.zeros_like(matrix, dtype=np.uint8)
    for i, level in enumerate(quantization_levels):
        lower_bound = int(i * level_width)
        upper_bound = int((i + 1) * level_width)
        mask = np.logical_and(matrix >= lower_bound, matrix < upper_bound)
        quantized_matrix[mask] = level

    return quantized_matrix

def huffman_encode(data, huffman_mapping):
    encoded_data = "".join(huffman_mapping[value] for value in data.flatten())
    return encoded_data

def huffman_decode(encoded_data, huffman_tree, original_shape):
    decoded_data = []
    current_node = huffman_tree

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.value is not None:
            decoded_data.append(current_node.value)
            current_node = huffman_tree

    return np.array(decoded_data).reshape(original_shape)

def calculate_compression_ratio(average_length):
        # Calculate compression ratio
        compression_ratio = 8 / average_length

        return compression_ratio


if image is not None:
    cropped_image = image

    color_cropped_image = cropped_image[:, :, 2]   #2 for red

    quantization_levels = [16, 48, 80, 112, 144, 176, 208, 240]

    # Quantize the matrix
    quantized_img_red = quantize_matrix(color_cropped_image, quantization_levels)

   # Flatten the quantized matrix
    flat_image = quantized_img_red.flatten()

    # Create a frequency dictionary
    frequency_dict = defaultdict(int)
    for value in flat_image:
        frequency_dict[value] += 1

    # Generate Huffman codes
    codebook = huffman.codebook(frequency_dict.items())
    print("Huffman codebook")
    print(codebook)

    #for full image

    blue_image = image[:, :, 0]  # 0 for blue
    green_image = image[:, :, 1]  # 1 for green
    red_image = image[:, :, 2]  # 2 for red

    blue_full_img = quantize_matrix(blue_image, quantization_levels)
    green_full_img = quantize_matrix(green_image, quantization_levels)
    red_full_img = quantize_matrix(red_image, quantization_levels)

    # Encode each color channel using Huffman codes
    red_encoded_full_image = huffman_encode(red_full_img, codebook)
    green_encoded_full_image = huffman_encode(green_full_img, codebook)
    blue_encoded_full_image = huffman_encode(blue_full_img,codebook)

    with open("Red compressed data inbuilt huffman.txt", "w") as text_file:
        text_file.write(red_encoded_full_image)

    with open("Green compressed data inbuilt huffman.txt", "w") as text_file:
        text_file.write(green_encoded_full_image)

    with open("Blue compressed data inbuilt huffman.txt", "w") as text_file:
        text_file.write(blue_encoded_full_image)

    print("Encoded text files saved")

    '''
    Calculate average length and compression ratio for cropped image from inbuild huffman codebook
    '''
    # cropped_prob = [(208, 0.78125), (176, 0.1484375), (240, 0.0234375), (48, 0.015625), (80, 0.015625), (16, 0.0078125),(112, 0.00390625), (144, 0.00390625)]

    # Average lengths
    average_length_cropped = sum(prob * len(codebook[symbol]) for symbol, prob in cropped_prob)
    print(f"Average length of the cropped image Huffman inbuilt: {average_length_cropped} bits per symbol")

    # compression ratios
    compression_ratio_cropped = calculate_compression_ratio(average_length_cropped)
    print("Compression Ratio of cropped image Huffman inbuilt:", compression_ratio_cropped)

    '''
     Calculate average length and compression ratio for full image from inbuild huffman codebook
     '''

    full_img_prob = [(240, 0.3627345679012346), (48, 0.21556481481481482), (208, 0.10142592592592592),
                     (80, 0.10026388888888889), (16, 0.08940895061728395), (176, 0.05304475308641975),
                     (112, 0.0424212962962963), (144, 0.0351358024691358)]

    average_length_full = sum(prob * len(codebook[symbol]) for symbol, prob in full_img_prob)
    print(f"Average length of the full image Huffman inbuilt: {average_length_full} bits per symbol")

    compression_ratio_full = calculate_compression_ratio(average_length_full)
    print("Compression Ratio of full image Huffman inbuilt:", compression_ratio_full)



