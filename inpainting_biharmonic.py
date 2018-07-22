import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace
import skimage
from skimage.measure import label, regionprops

def numbergrid(mask):
    n = np.sum(mask)
    G1 = np.zeros_like(mask, dtype=np.uint32)
    G1[mask] = np.arange(1, n+1)
    return G1

def delsq_laplacian(G):
    [m, n] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 4 * np.ones(N)
    #for all four neighbor of center points
    for offset in [-1, m, 1, -m]:
        #indices of all possible neighbours in sparse matrix
        Q = G1[p+offset]
        #filter inner indices
        q = np.where(Q)[0]
        #generate neighbour coordinate
        i = np.concatenate([i, G1[p[q]]-1])
        j = np.concatenate([j, Q[q]-1])
        s = np.concatenate([s, -np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i,j)), (N,N))
    return sp

def delsq_bilaplacian(G):
    [n, m] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 20 * np.ones(N)
    #for all four neighbor of center points
    coeffs  = np.array([1, 2, -8, 2, 1, -8, -8, 1, 2, -8, 2, 1])
    offsets = np.array([-2*m, -m-1, -m, -m+1, -2, -1, 1, 2, m-1, m, m+1, 2*m])
    for coeff, offset in zip(coeffs, offsets):
        #indices of all possible neighbours in sparse matrix
        Q = G1[p+offset]
        #filter inner indices
        q = np.where(Q)[0]
        #generate neighbour coordinate
        i = np.concatenate([i, G1[p[q]]-1])
        j = np.concatenate([j, Q[q]-1])
        s = np.concatenate([s, coeff*np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i,j)), (N,N))
    return sp

def generate_stencials():
    stencils = []
    for i in range(5):
        for j in range(5):
            A = np.zeros((5, 5))
            A[i,j]=1
            S = laplace(laplace(A))
            x_range = np.array([i-2, i+3]).clip(0,5)
            y_range = np.array([j-2, j+3]).clip(0,5)
            S = S[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            stencils.append(S)

    return stencils

def _inpaint_biharmonic_single_channel(mask, out, limits):
    # Initialize sparse matrices
    matrix_unknown = sparse.lil_matrix((np.sum(mask), out.size))
    matrix_known = sparse.lil_matrix((np.sum(mask), out.size))

    # Find indexes of masked points in flatten array
    mask_i = np.ravel_multi_index(np.where(mask), mask.shape)

    G = numbergrid(mask)
    L = delsq_bilaplacian(G)
    out[mask] = 0
    B = -laplace(laplace(out))
    plt.imshow(B, cmap='gray')
    plt.show()
    b = B[mask]
    result = spsolve(L, b)
    # Handle enormous values
    result = np.clip(result, *limits)
    result = result.ravel()
    out[mask] = result
    return out

def dilate_rect(rect, d, nd_shape):
    rect[0:2] = (rect[0:2] - d).clip(min = 0)
    rect[2:4] = (rect[2:4] + d).clip(max = nd_shape)
    return rect

def k_inpaint_biharmonic(image, mask, multichannel=False):
    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')

    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)
    mask = mask.astype(np.bool)

    # Split inpainting mask into independent regions
    kernel = ndi.morphology.generate_binary_structure(mask.ndim, 1)
    mask_dilated = ndi.morphology.binary_dilation(mask, structure=kernel)
    mask_labeled, num_labels = label(mask_dilated, return_num=True)
    mask_labeled *= mask
    if not multichannel:
        image = image[..., np.newaxis]

    out = np.copy(image)

    props = regionprops(mask_labeled)
    comp_out_imgs = []
    comp_masks = []
    for i in range(num_labels):
        rect = np.array(props[i].bbox)
        rect = dilate_rect(rect, 2, image.shape[:2])
        out_sub_img = out[rect[0]:rect[2], rect[1]:rect[3], :]
        comp_mask   = mask[rect[0]:rect[2], rect[1]:rect[3]]
        # plt.subplot(121), plt.imshow(comp_mask)
        # plt.subplot(122), plt.imshow(out_sub_img)
        # plt.show()
        comp_out_imgs.append(out_sub_img)
        comp_masks.append(comp_mask)

    for idx_channel in range(image.shape[-1]):
        known_points = image[..., idx_channel][~mask]
        limits = (np.min(known_points), np.max(known_points))
        for i in range(num_labels):
            _inpaint_biharmonic_single_channel(comp_masks[i], comp_out_imgs[i][..., idx_channel], limits)

    if not multichannel:
        out = out[..., 0]

    return out

image_orig = data.astronaut()[0:200, 0:200]

# Create mask with three defect regions: left, middle, right respectively
mask = np.zeros(image_orig.shape[:-1])
mask[160:180, 70:155] = 1

# Defect image over the same region in each color channel
image_defect = image_orig.copy()
for layer in range(image_defect.shape[-1]):
    image_defect[np.where(mask)] = 0

image_result = k_inpaint_biharmonic(image_defect, mask, multichannel=True)

fig, axes = plt.subplots(ncols=2, nrows=2)
ax = axes.ravel()

ax[0].set_title('Original image')
ax[0].imshow(image_orig)

ax[1].set_title('Mask')
ax[1].imshow(mask, cmap=plt.cm.gray)

ax[2].set_title('Defected image')
ax[2].imshow(image_defect)

ax[3].set_title('Inpainted image')
ax[3].imshow(image_result)

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()