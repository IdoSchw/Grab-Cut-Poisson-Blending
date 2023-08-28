import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import argparse

im_tgt_height = -1
im_tgt_width = -1
N = -1

# setting global dimensions according to the target
def set_global_dimensions(shape):
    global im_tgt_height
    global im_tgt_width
    global N

    im_tgt_height = shape[0]
    im_tgt_width = shape[1]
    N = im_tgt_height * im_tgt_width


# A function to pad im_src and im_mask to be the size of im_tgt
def pad_im(im):
    # Get the size of the image
    im_height = im.shape[0]
    im_width = im.shape[1]

    # Calculate the difference in height and width
    diff_in_height = im_tgt_height - im_height
    diff_in_width = im_tgt_width - im_width

    # Calculate the top, bottom, left, and right padding sizes
    # we enforce top + bottom = diff_in_height so that height(im) will be equal to height(im_tgt)
    top = diff_in_height // 2
    bottom = diff_in_height - top
    left = diff_in_width // 2
    right = diff_in_width - left
    # Create a border with the calculated padding sizes
    bordered_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return bordered_im

# a function that sets the pixel's values according to the arguments given
def set_pixels(pix, A, middle, edges):
    A[pix, pix] = middle
    if pix + 1 < N:
        A[pix, pix + 1] = edges
    if pix - 1 > 0:
        A[pix, pix - 1] = edges
    if pix + im_tgt_width < N:
        A[pix, pix + im_tgt_width] = edges
    if pix - im_tgt_width > 0:
        A[pix, pix - im_tgt_width] = edges

# if the pixel is outside of the mask, then when computing the gradient:
# the middle pixel should be 1, and the rest around it should be 0
def set_pixel_outside_mask(pix, A):
    set_pixels(pix, A, 1, 0)

# if the pixel is inside of the mask, then when computing the gradient:
# the middle pixel should be -4, and the rest around it should be 1
def set_pixel_inside_mask(pix, A):
    set_pixels(pix, A, -4, 1)


# creating the matrix of the gradients
def create_sparse_coefficients_matrix(im_mask):
    # a better representation of the matrix, because it's sparsed(and has many 0's)
    A = sp.lil_matrix((N, N))
    for row in range(0, im_tgt_height):
        for col in range(0, im_tgt_width):
            pix = col + row * im_tgt_width
            # setting the value of the gradient based on the pixel's type - inside or outside the mask
            if im_mask[row, col] == 0:
                set_pixel_outside_mask(pix, A)
            else:
                set_pixel_inside_mask(pix, A)
    return A.tocsc()


# setting the color where mask == 0 so that vector[pix] = target[pix]
def set_outside_of_mask_color_to_tgt(b_vector, mask_arr, tgt_color_arr):
    for i in range(len(mask_arr) - 1):
        if mask_arr[i] == 0:
            b_vector[i] = tgt_color_arr[i]
    return b_vector

# for each color:
# checking it's values in the source and the target
# and if it's for a pixel outside of the mask, then it needs to be
# set to be as the target's value in that pixel
def divergence_by_color(color, A, im_src, im_tgt, im_mask):

    #####################
    # color = 0 => Red
    # color = 1 => Green
    # color = 2 => Blue
    #####################

    # we want a vector of an only one color
    src_color = im_src[:, :, color]
    tgt_color = im_tgt[:, :, color]

    # representing the matrices as arrays
    src_color_arr = src_color.flatten()
    tgt_color_arr = tgt_color.flatten()
    mask_arr = im_mask.flatten()

    # b_vector contains the value of the gradient according to the color
    b_vector = A.dot(src_color_arr)
    # setting the color where mask = 0 so that vector[pix] = target[pix]
    b_vector = set_outside_of_mask_color_to_tgt(b_vector, mask_arr, tgt_color_arr)

    return b_vector


# reshaping a vector to be of matrix of size im_tgt_height x im_tgt_width
def reshape_vector(vector):
    return vector.reshape((im_tgt_height, im_tgt_width))

# checking each value in the vector is between 0 and 255
# and that each value is of type uint8(for ndarray)
def check_range_and_type(vector):
    for i in range(len(vector)):
        if vector[i] < 0:
            vector[i] = 0
        if vector[i] > 255:
            vector[i] = 255
    return vector.astype('uint8')


# creating the final image from the 3 color matrices
def create_blended_mat(R_mat, G_mat, B_mat):
    im_blend = np.zeros_like(im_tgt)
    # im_blend[row, col, color] = color_mat[row, col]
    im_blend[..., 0] = R_mat
    im_blend[..., 1] = G_mat
    im_blend[..., 2] = B_mat
    return im_blend


# setting the vectors of each color (for the source and the target)
# according to the mask
def set_vectors_of_RGB(A, im_src, im_tgt, im_mask):
    R_vector = divergence_by_color(0, A, im_src, im_tgt, im_mask)
    G_vector = divergence_by_color(1, A, im_src, im_tgt, im_mask)
    B_vector = divergence_by_color(2, A, im_src, im_tgt, im_mask)
    return R_vector, G_vector, B_vector

# solving the equations Ax = b for each color
def solve_RGB_vectors(A, R_vector, G_vector, B_vector):
    R_solved = spsolve(A, R_vector)
    G_solved = spsolve(A, G_vector)
    B_solved = spsolve(A, B_vector)
    return R_solved, G_solved, B_solved

# checking each value in the vector is between 0 and 255
def check_range_for_RGB(R_solved, G_solved, B_solved):
    R_solved = check_range_and_type(R_solved)
    G_solved = check_range_and_type(G_solved)
    B_solved = check_range_and_type(B_solved)
    return R_solved, G_solved, B_solved

# reshaping the color-vectors to a matrix
def reshape_RGB_vectors(R_solved, G_solved, B_solved):
    R_mat = reshape_vector(R_solved)
    G_mat = reshape_vector(G_solved)
    B_mat = reshape_vector(B_solved)
    return R_mat,G_mat, B_mat


def poisson_blend(im_src, im_tgt, im_mask, center):
    # A is the sparse coefficients matrix
    A = create_sparse_coefficients_matrix(im_mask)

    # setting the vectors of each color for the source and the target
    # according to the mask
    R_vector, G_vector, B_vector = set_vectors_of_RGB(A, im_src, im_tgt, im_mask)

    # solving the equations to each color
    R_solved, G_solved, B_solved = solve_RGB_vectors(A, R_vector, G_vector, B_vector)

    # checking the range of each color is between 0 and 255
    R_solved, G_solved, B_solved = check_range_for_RGB(R_solved, G_solved, B_solved)

    # reshaping the vectors into a matrix, shape(color_mat) = shape(target)
    R_mat, G_mat, B_mat = reshape_RGB_vectors(R_solved, G_solved, B_solved)

    # creating the final img from the 3 color matrices
    im_blend = create_blended_mat(R_mat, G_mat, B_mat)

    return im_blend

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()
    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    cv2.waitKey(0)
    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))
    set_global_dimensions(im_tgt.shape)
    padded_src = pad_im(im_src)
    padded_mask = pad_im(im_mask)

    im_clone = poisson_blend(padded_src, im_tgt, padded_mask, center)

    im_clone = cv2.resize(im_clone, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()