import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import heapq

'''
From this code,
 Make the huffman tree for decoding
 Decompress text files created from the inbuilt huffman codebook
'''

full_image = cv.imread('Sample1.jpg')   # Read the image

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


if full_image is not None:

    red_full_image = full_image[:, :, 2]   # Take the red value (2) for build the huffman code

    '''
    Since the cropped image has only 2 quantized values (208,176),
    i have assigned some pixel values in cropped image only for build the huffman code
    '''

    quantization_levels = [16, 48, 80, 112, 144, 176, 208, 240]  # Quantization levels

    # Quantize the matrix
    quantized_img_red = quantize_matrix(red_full_image, quantization_levels)

    # Flatten the quantized image array to a 1D array
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

    # Build Huffman tree
    huffman_tree = huffman_tree_build(sorted_probabilities)

    # Generate Huffman codes
    huffman_mapping = huffman_codes(huffman_tree)

    print("Huffman Mapping:")
    print(huffman_mapping)

    # Take the separated color channels of full image
    blue_image = full_image[:, :, 0]  # 0 for blue
    green_image = full_image[:, :, 1]  # 1 for green
    red_image = full_image[:, :, 2]  # 2 for red

    # Quantize each full images
    blue_full_img = quantize_matrix(blue_image, quantization_levels)
    green_full_img = quantize_matrix(green_image, quantization_levels)
    red_full_img = quantize_matrix(red_image, quantization_levels)

    '''
    Decompressing the text files of build in huffman codebook in python
    '''
    # Read the saved text files , encoded by build in function in python
    content_red_inbuilt = read_compressed_data("Red compressed data inbuilt huffman.txt")
    content_green_inbuilt = read_compressed_data("Green compressed data inbuilt huffman.txt")
    content_blue_inbuilt = read_compressed_data("Blue compressed data inbuilt huffman.txt")

    # Decode the data of the text files (build in huffman codebook)
    red_decoded_txt_inbuilt = huffman_decode(content_red_inbuilt, huffman_tree, red_full_img.shape)
    green_decoded_txt_inbuilt = huffman_decode(content_green_inbuilt, huffman_tree, green_full_img.shape)
    blue_decoded_txt_inbuilt = huffman_decode(content_blue_inbuilt, huffman_tree, blue_full_img.shape)

    # Combine decoded data of all three channels and construct the colored full image(buil in huffman codebook)
    decoded_image_from_txt_inbuilt = np.stack([blue_decoded_txt_inbuilt, green_decoded_txt_inbuilt, red_decoded_txt_inbuilt], axis=-1)

    # Showing the decompressed image of encoded by the build in function in python
    plt.imshow(cv.cvtColor(decoded_image_from_txt_inbuilt, cv.COLOR_BGR2RGB))
    plt.title("Decompressed image using inbuilt Huffman code in python")
    plt.axis('off')
    plt.show()

    # # Use for save image
    #
    # cv.imwrite("Quantized_full_image.jpg", red_full_img)
    # cv.imwrite("Decompressed_image_buildin_huffman.jpg", decoded_image_from_txt_inbuilt)






