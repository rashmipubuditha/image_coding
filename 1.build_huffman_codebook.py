import heapq
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
From this code,
 create the huffman codebook using cropped image
 Encoded the full image and save compressed text files
 Decode those text files and create the decompressed image
 Calculate entropy, psnr, average length , compression ratios using created huffman algorith
'''

image = cv.imread('Sample1.jpg')   # Read the image
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original image")
plt.axis('off')
plt.show()

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

def calculate_probabilities(data):
    unique_values, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return zip(unique_values, probabilities)

def huffman_tree_build(probabilities):
    heap = [Node(value, freq) for value, freq in probabilities]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def huffman_codes(node, code="", mapping=None):
    if mapping is None:
        mapping = {}

    if node is not None:
        if node.value is not None:
            mapping[node.value] = code
        huffman_codes(node.left, code + "0", mapping)
        huffman_codes(node.right, code + "1", mapping)

    return mapping

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

def read_compressed_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def calculate_entropy(probability_distribution):
    probabilities = np.array([probability for _, probability in probability_distribution])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_psnr(original_image, compressed_image, max_pixel_value=255):
    mse = np.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinity when MSE is zero
    else:
        psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)
        return psnr_value

def calculate_compression_ratio(average_length):

        # Calculate compression ratio
        compression_ratio = 8 / average_length

        return compression_ratio


if image is not None:

    # Pixel position according to my E number (E/18/268) x: 2*60 = 120 , y: 68*4 = 272
    y, x = 272, 120
    crop_size = 16
    cropped_image = image[y:y+crop_size, x:x+crop_size]

    color_cropped_image = cropped_image[:, :, 2]   # Take the red value (2) for build the huffman code

    '''
    Since the cropped image has only 2 quantized values (208,176),
    i have assigned some pixel values in cropped image only for build the huffman code
    '''

    positions_and_values = {(0, 1): 20,(1, 1): 20,(0,2 ): 45,(1,2 ): 45,(3,2 ): 45,(2,2 ): 45,(0, 3): 83,(1, 3): 83,(2, 3): 83,(3, 3): 83,(0, 4): 120,(0, 5): 145,(0, 6): 245,(0, 7): 245,(0, 8): 245,(0, 9): 245,(0, 10): 245,(0, 11): 245}

    for position, value in positions_and_values.items():
        color_cropped_image[position] = value

    quantization_levels = [16, 48, 80, 112, 144, 176, 208, 240]  # Quantization levels


    # Quantize the rearranged cropped image
    quantized_img_red = quantize_matrix(color_cropped_image, quantization_levels)

    # Flatten the rearranged quantized image array to a 1D array
    flat_image = quantized_img_red.flatten()

    # Calculate the histogram of the quantized image
    hist, bins = np.histogram(flat_image, bins=256, range=(0, 256), density=True)

    # Plot the histogram
    plt.bar(bins[:-1], hist, width=10.0, edgecolor='black')
    plt.title('Probability Distribution (Histogram) of Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability')
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate probabilities
    probabilities = list(calculate_probabilities(flat_image))

    # Sort the list based on the second element of each tuple (the probability)
    # sorted_probabilities = sorted(probabilities, key=lambda x: x[1], reverse=True)

    # Display the sorted list
    print("Probability of each symbol distribution:")
    print(sorted_probabilities)

    # find entropy
    entropy_rearranged_cropped = calculate_entropy(sorted_probabilities)
    print(f"Entropy of the source: {entropy_rearranged_cropped}")

    # Build Huffman tree
    huffman_tree = huffman_tree_build(sorted_probabilities)

    # Generate Huffman codes
    huffman_mapping = huffman_codes(huffman_tree)

    print("Huffman Mapping:")
    print(huffman_mapping)

    '''
    Encode the cropped image for Red, Green, Blue color channels
    '''
    # Take the cropped images in 3 colors
    blue_cropped_image = cropped_image[:, :, 0]  # 0 for blue
    green_cropped_image = cropped_image[:, :, 1]  # 1 for green
    red_cropped_image = cropped_image[:, :, 2]  # 2 for red

    # Quantize each cropped images into quantized levels
    blue_quantized_img = quantize_matrix(blue_cropped_image, quantization_levels)
    green_quantized_img = quantize_matrix(green_cropped_image, quantization_levels)
    red_quantized_img = quantize_matrix(red_cropped_image, quantization_levels)

    # Encode each color channels of cropped images using Huffman codes
    red_encoded_data = huffman_encode(red_quantized_img, huffman_mapping)
    green_encoded_data = huffman_encode(green_quantized_img, huffman_mapping)
    blue_encoded_data = huffman_encode(blue_quantized_img, huffman_mapping)

    '''
    Encode the full image into Red, Green, Blue color channels
    '''
    # Take the separated color channels of full image
    blue_image = image[:, :, 0]  # 0 for blue
    green_image = image[:, :, 1]  # 1 for green
    red_image = image[:, :, 2]  # 2 for red

    # Quantize each full images
    blue_full_img = quantize_matrix(blue_image, quantization_levels)
    green_full_img = quantize_matrix(green_image, quantization_levels)
    red_full_img = quantize_matrix(red_image, quantization_levels)

    # Encode each color channels of full image using Huffman codes
    red_encoded_full_image = huffman_encode(red_full_img, huffman_mapping)
    green_encoded_full_image = huffman_encode(green_full_img, huffman_mapping)
    blue_encoded_full_image = huffman_encode(blue_full_img, huffman_mapping)

    # Compressed encoded three color channels as text files
    with open("Red compressed data.txt", "w") as text_file:  # Red
        text_file.write(red_encoded_full_image)

    with open("Green compressed data.txt", "w") as text_file:  # Green
        text_file.write(green_encoded_full_image)

    with open("Blue compressed data.txt", "w") as text_file:  # Blue
        text_file.write(blue_encoded_full_image)

    '''
    Decompressing the text files of constructed huffman code
    '''
    # Read the saved text files , encoded by constructed huffman code
    content_red = read_compressed_data("Red compressed data.txt")
    content_green = read_compressed_data("Green compressed data.txt")
    content_blue = read_compressed_data("Blue compressed data.txt")

    # Decode the data of the text file
    red_decoded_data_txt = huffman_decode(content_red, huffman_tree, red_full_img.shape)
    green_decoded_data_txt = huffman_decode(content_green, huffman_tree, green_full_img.shape)
    blue_decoded_data_txt = huffman_decode(content_blue, huffman_tree, blue_full_img.shape)

    # Combine decoded data of all three channels and construct the colored full image
    decoded_image_from_txt = np.stack([blue_decoded_data_txt, green_decoded_data_txt, red_decoded_data_txt], axis=-1)

    # Showing the decompressed image of encoded by the constructed huffman code
    # plt.imshow(cv.cvtColor(decoded_image_from_txt, cv.COLOR_BGR2RGB))
    # plt.title("Decompressed image")
    # plt.axis('off')
    # plt.show()

    '''
    Find the PSNR of the image
    '''

    # find PSNR original image
    psnr = calculate_psnr(image, image)
    print(f"PSNR original image: {psnr} dB")

    # find PSNR decompressed image
    psnr_decompressed = calculate_psnr(image, decoded_image_from_txt)
    print(f"PSNR decompressed image: {psnr_decompressed} dB")

    # Use for save the images

    # cv.imwrite("Cropped_image.jpg", cropped_image)
    # cv.imwrite("Quantized_cropped_image_remake.jpg", quantized_img_red)
    # cv.imwrite("Quantized_cropped_image.jpg", red_quantized_img)
    # cv.imwrite("Quantized_full_image.jpg", red_full_img)

    #Calculate entropy for full image
    red_original_flat_image = image[:, :, 2].flatten()  # flat the red quantized image
    probability_red_image = list(calculate_probabilities(red_original_flat_image))
    entropy_full_image = calculate_entropy(probability_red_image)
    print(f"Entropy of the original image: {entropy_full_image}")

    #calculate entropy of cropped image
    red_cropped_flat_image = color_cropped_image.flatten()  # flat the red quantized image
    probability_red_cropped_image = list(calculate_probabilities(red_cropped_flat_image))
    entropy_cropped_image = calculate_entropy(probability_red_cropped_image)
    print(f"Entropy of the cropped image: {entropy_cropped_image}")

    # Calculate entropy for decompress image
    red_decompress_flat_image = decoded_image_from_txt[:, :, 2].flatten()  # flat the red quantized image
    probability_red_decompress_image = list(calculate_probabilities(red_decompress_flat_image))
    entropy_decompress_image = calculate_entropy(probability_red_decompress_image)
    print(f"Entropy of the decompress image: {entropy_decompress_image}")

    '''
    Calculate average length and compression ratio for cropped image from my huffman mapping
    '''
    # Calculate the average length of the cropped image
    average_length_cropped = sum(prob * len(huffman_mapping[symbol]) for symbol, prob in sorted_probabilities)
    print(f"Average length of the cropped image: {average_length_cropped} bits per symbol")

    # Calculate compression ratio of cropped image
    compression_ratio_cropped = calculate_compression_ratio(average_length_cropped)
    print("Compression Ratio of cropped image:", compression_ratio_cropped)

    '''
    Calculate average length and compression ratio for full image from my huffman mapping
    '''
    # full_img_prob = [(240, 0.3627345679012346), (48, 0.21556481481481482), (208, 0.10142592592592592),(80, 0.10026388888888889), (16, 0.08940895061728395), (176, 0.05304475308641975),(112, 0.0424212962962963), (144, 0.0351358024691358)]

    average_length_full = sum(prob * len(huffman_mapping[symbol]) for symbol, prob in full_img_prob)
    print(f"Average length of the full image: {average_length_full} bits per symbol")

    compression_ratio_full = calculate_compression_ratio(average_length_full)
    print("Compression Ratio of full image:", compression_ratio_full)

else:
    print(f"Error: Image not found: {image}")
