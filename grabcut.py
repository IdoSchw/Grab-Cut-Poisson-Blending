import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
H = None
W = None
TOTAL_PIXELS = None
n_links = []
edges_weights = []
beta = None
K = None
prev_energy = float('inf')
comp_map = None
p_to_v_dict = None
v_to_p_dict = None
EPSILON = 10 ** (-5)
EPSILON_MAT = EPSILON * np.eye(3)

np.random.seed(0)
energy_delta_threshold = 3000
N_COMPONENTS = 2


# Receives The img array and updates globals
# with the height, width and the number of pixels of the image
def init_img_globals(img):
    global H
    global W
    global TOTAL_PIXELS
    H, W = len(img), len(img[0])
    TOTAL_PIXELS = H * W


# Initializes 2 dim array for the component mapping
# (in the dimensions of the mask)
def init_comp_map():
    global comp_map
    comp_map = np.zeros([H, W], dtype=np.uint8)


# Receives img, mask, GMMs and the number of components
# in each GMM, and updates the component mapping.
# Each entry of the component mapping is the component in which
# the current pixel is in (for foreground pixel we have a shift of n_comp).
def update_comp_map(img, mask, bgGMM, fgGMM, n_comp):
    bg_ind = np.where(np.logical_or(mask == GC_BGD, mask == GC_PR_BGD))
    fg_ind = np.where(np.logical_or(mask == GC_FGD, mask == GC_PR_FGD))
    n_bg, n_fg = len(bg_ind[0]), len(fg_ind[0])
    bg_comp_options = np.zeros([n_comp, n_bg], dtype=np.float64)
    fg_comp_options = np.zeros([n_comp, n_fg], dtype=np.float64)
    for comp_ind in range(n_comp):
        bg_comp_options[comp_ind] = calc_pdf(bgGMM, bg_ind, comp_ind, img, n_bg)
        fg_comp_options[comp_ind] = calc_pdf(fgGMM, fg_ind, comp_ind, img, n_fg)
    comp_map[bg_ind] = np.argmax(bg_comp_options, axis=0)
    comp_map[fg_ind] = np.argmax(fg_comp_options, axis=0) + n_comp


# Receives a GMM, the indices of the pixels of all its components,
# an index of component, the img and the number of pixels in background
# or foreground. Returns for each pixel in the relevant ground (bg/fg)
# an array of the probability of each pixel to be in the specific component
# for which the index is one of the arguments
def calc_pdf(GMM, ground_ind, comp_ind, img, n):
    if GMM.weights_[comp_ind] <= 0:
        return np.zeros(n)
    s = GMM.covariances_[comp_ind]
    det_s = np.linalg.det(s)
    inv_s = np.linalg.inv(s)
    x = img[ground_ind]
    mu = GMM.means_[comp_ind]
    nom = np.exp(-0.5 * np.sum((x-mu) * (np.dot(inv_s, (x-mu).T)).T, axis=1))
    denom = np.sqrt((2 * np.pi)**3) * det_s
    return nom / denom


# Receives the GMMs, img and number of components and updates
# all the components' mean values
def update_means(bgGMM, fgGMM, img, n_comp):
    for comp_ind in range(n_comp):
        bgGMM.means_[comp_ind] = np.mean(img[comp_map == comp_ind], axis=0)
        fgGMM.means_[comp_ind] = np.mean(img[comp_map == comp_ind + n_comp], axis=0)


# Receives the GMMs, img and number of components and updates
# all the components' covariances
def update_covs(bgGMM, fgGMM, img, n_comp):
    for comp_ind in range(n_comp):
        bgGMM.covariances_[comp_ind] = get_invertible_cov(np.cov(img[comp_map == comp_ind], rowvar=False))
        fgGMM.covariances_[comp_ind] = get_invertible_cov(np.cov(img[comp_map == comp_ind + n_comp], rowvar=False))


# Receives a covariances matrix which may be not invertible and
# returns close covariances matrix which is invertible
def get_invertible_cov(mat):
    while np.linalg.det(mat) <= 0:
        mat += EPSILON_MAT
    return mat


# Receives the GMMs, img and number of components and updates
# all the components' weights
def update_weights(bgGMM, fgGMM, mask, n_comp):
    n_bg = np.count_nonzero(np.logical_or(mask==GC_BGD, mask==GC_PR_BGD))
    n_fg = np.count_nonzero(np.logical_or(mask==GC_FGD, mask==GC_PR_FGD))
    for comp_ind in range(n_comp):
        bgGMM.weights_[comp_ind] = np.count_nonzero(comp_map == comp_ind) / n_bg
        fgGMM.weights_[comp_ind] = np.count_nonzero(comp_map == comp_ind + n_comp) / n_fg


# Updates 2 dictionaries: one from pixel indices to vertex at the graph
# and the second one is the other direction
def init_dicts():
    global p_to_v_dict
    global v_to_p_dict
    pixel_v_list = [(i, j) for i in range(H) for j in range(W)] + ["bg", "fg"]
    p_to_v_dict = {pixel: v for v, pixel in enumerate(pixel_v_list)}
    v_to_p_dict = {v: pixel for v, pixel in enumerate(pixel_v_list)}


# Initializes the graph of the algorithm, adds n-links
# and returns the graph
def init_graph():
    g = ig.Graph()
    if not p_to_v_dict:
        init_dicts()
    g.add_vertices(TOTAL_PIXELS + 2)
    g = add_n_edges(g)
    return g


# Receives a graph, adds the n-links and returns it
def add_n_edges(g):
    global n_links
    global edges_weights
    if len(n_links) == 0:
        vertical = [((i + 1, j), (i, j)) for i in range(H - 1) for j in range(W)]
        horizontal = [((i, j + 1), (i, j)) for i in range(H) for j in range(W - 1)]
        left_down = [((i + 1, j - 1), (i, j)) for i in range(H - 1) for j in range(1, W)]
        right_down = [((i + 1, j + 1), (i, j)) for i in range(H-1) for j in range(W - 1)]
        update_beta(img, vertical, horizontal, left_down, right_down)
        n_links = vertical + horizontal + left_down + right_down
    g.add_edges([(p_to_v_dict[m], p_to_v_dict[n]) for m,n in n_links])
    edges_weights = [calc_n_link_weight(m, n) for m, n in n_links]
    return g


# Receives img and all the options of n-links edges
# and updates the beta global (from the document)
def update_beta(img, vertical, horizontal, left_down, right_down):
    global beta
    v = [(img[h1][w1] - img[h2][w2]) ** 2 for ((h1, w1), (h2, w2)) in vertical]
    h = [(img[h1][w1] - img[h2][w2]) ** 2 for ((h1, w1), (h2, w2)) in horizontal]
    ld = [(img[h1][w1] - img[h2][w2]) ** 2 for ((h1, w1), (h2, w2)) in left_down]
    rd = [(img[h1][w1] - img[h2][w2]) ** 2 for ((h1, w1), (h2, w2)) in right_down]
    s = np.sum(v) + np.sum(h) + np.sum(ld) + np.sum(rd)
    n = len(v) + len(h) + len(ld) + len(rd)
    beta = 1/(2 * (s / n))


# Receives 2 pixels (indices) and returns the n-link weight
# between them (assuming there is one such)
def calc_n_link_weight(m, n):
    # m and n are both 2-elements tuples
    dist_m_n = np.sqrt((m[0] - n[0])**2 + (m[1] - n[1])**2)
    dist_zm_zn_sq = np.sum((img[m] - img[n]) ** 2, axis=0)
    return (50 / dist_m_n) * (np.exp(-beta * dist_zm_zn_sq))


# Receives a graph, edges list and weights list,
# adds the edges to the graph and adds to the global edges_weights list
# the weights of the edges
def update_graph_with_edges(g, edges, weights):
    global edges_weights
    g.add_edges(edges)
    edges_weights += weights
    return g


# Receives the graph, img, mask and the 2 GMMs and
# returns the graph with the t-links edges
def add_t_edges(g, img, mask, bgGMM, fgGMM):
    global K
    if not K:
        K = calc_K()
    D_fore = calc_D(fgGMM, img, mask)
    D_back = calc_D(bgGMM, img, mask)
    t_f_b_links = [(p_to_v_dict[(i, j)], p_to_v_dict["bg"]) for i, j in zip(*np.where(mask == GC_FGD)[:2])]
    g = update_graph_with_edges(g, t_f_b_links, [0 for _ in range(len(t_f_b_links))])
    t_f_f_links = [(p_to_v_dict[(i, j)], p_to_v_dict["fg"]) for i, j in zip(*np.where(mask == GC_FGD)[:2])]
    g = update_graph_with_edges(g, t_f_f_links, [K for _ in range(len(t_f_f_links))])
    t_b_b_links = [(p_to_v_dict[(i, j)], p_to_v_dict["bg"]) for i, j in zip(*np.where(mask == GC_BGD)[:2])]
    g = update_graph_with_edges(g, t_b_b_links, [K for _ in range(len(t_b_b_links))])
    t_b_f_links = [(p_to_v_dict[(i, j)], p_to_v_dict["fg"]) for i, j in zip(*np.where(mask == GC_BGD)[:2])]
    g = update_graph_with_edges(g, t_b_f_links, [0 for _ in range(len(t_b_f_links))])
    t_u_f_links = [(p_to_v_dict[(i, j)], p_to_v_dict["fg"]) for i, j in zip(*np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))[:2])]
    g = update_graph_with_edges(g, t_u_f_links, list(D_back))
    t_u_b_links = [(p_to_v_dict[(i, j)], p_to_v_dict["bg"]) for i, j in zip(*np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))[:2])]
    g = update_graph_with_edges(g, t_u_b_links, list(D_fore))
    return g


# computing an upper bound for the edges' weights of a pixel neighbours
def calc_K():
    max_weight = max(edges_weights)
    # upper bound because it's an un-directional graph, and 4 neighbors => 2 * 4 * weight
    return 8 * max_weight


# Sub-computation of the D-value (D_fore/D_back) from the document
def calc_left_side(gmm, comp):
    sig_determinant = np.linalg.det(gmm.covariances_[comp])
    return (gmm.weights_[comp]) / (np.sqrt(sig_determinant))


# Sub-computation of the D-value (D_fore/D_back) from the document
def calc_right_side(gmm, comp, img, mask):
    z_minus_mu = img[np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD)] - gmm.means_[comp]
    inv_cov = np.linalg.inv(gmm.covariances_[comp])
    product = np.einsum('ml,mn,nl->l', z_minus_mu.T, inv_cov, z_minus_mu.T)
    return np.exp(-0.5 * product)


# Sub-computation of the D-value (D_fore/D_back) from the document
def calc_inside_equation(gmm, comp, img, mask):
    left_side = calc_left_side(gmm, comp)
    right_side = calc_right_side(gmm, comp, img, mask)
    return left_side * right_side


# Receives a GMM, img and mask and
# returns the D-value (D_fore/D_back) from the document
def calc_D(gmm, img, mask):
    num_of_comp = len(gmm.weights_)
    sum_inside_equation = sum([calc_inside_equation(gmm, comp, img, mask) for comp in range(num_of_comp)])
    return -np.log(sum_inside_equation)

# Receives a mask and returns a mask in which all the entries
# which were "probably" become "sure"
def fix_mask(mask):
    mask[mask == GC_PR_BGD] = GC_BGD
    mask[mask == GC_PR_FGD] = GC_FGD
    return mask


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask, N_COMPONENTS)
    # I added the n_components argument like in the assignment.
    # Here this argument is assigned to be a global which is initailized
    # in the beginning of the code. I asked and was told it's ok to keep it in
    # such a way in the submission

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break
    mask = fix_mask(mask)
    # I added the last function to the grabcut to fix the "probably"
    # values of the mask to be "sure"

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components=5):
    # TODO: implement initalize_GMMs
    bgGMM = GaussianMixture(n_components).fit(img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)])
    fgGMM = GaussianMixture(n_components).fit(img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)])
    init_img_globals(img)
    init_comp_map()
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    n_comp = len(bgGMM.means_)
    update_comp_map(img, mask, bgGMM, fgGMM, n_comp)
    update_means(bgGMM, fgGMM, img, n_comp)
    update_covs(bgGMM, fgGMM, img, n_comp)
    update_weights(bgGMM, fgGMM, mask, n_comp)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0
    g = init_graph()
    g = add_t_edges(g, img, mask, bgGMM, fgGMM)
    cut = g.st_mincut(source=p_to_v_dict["fg"], target=p_to_v_dict["bg"], capacity=edges_weights)
    min_cut, energy = cut.partition, cut.value
    return min_cut, energy


def update_mask(mincut_sets, mask):
    if p_to_v_dict["bg"] in mincut_sets[0]:
        bg_set, fg_set = mincut_sets[0], mincut_sets[1]
    else:
        bg_set, fg_set = mincut_sets[1], mincut_sets[0]
    bg_set.remove(p_to_v_dict["bg"])
    fg_set.remove(p_to_v_dict["fg"])
    for v in bg_set:
        i, j = v_to_p_dict[v][0], v_to_p_dict[v][1]
        if mask[i][j] == GC_PR_FGD:
            mask[i][j] = GC_PR_BGD
    for v in fg_set:
        i, j = v_to_p_dict[v][0], v_to_p_dict[v][1]
        if mask[i][j] == GC_PR_BGD:
            mask[i][j] = GC_PR_FGD
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    global prev_energy
    if prev_energy - energy < energy_delta_threshold:
        return True
    prev_energy = energy
    return False


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation
    accuracy = np.sum(predicted_mask == gt_mask) / TOTAL_PIXELS
    jaccard_sim = np.sum(np.logical_and(predicted_mask, gt_mask)) / np.sum(np.logical_or(predicted_mask, gt_mask))
    return accuracy, jaccard_sim


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
