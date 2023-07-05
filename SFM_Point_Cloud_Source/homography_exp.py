"""

Sunny Karmur: Augmented Visuals with Homography


"""
import cv2
import numpy as np

def computeHomography(Us, Vs):
    """
    Estimate the homography matrix from given 4 points
    """
    # matrixA construction

    x1, y1 = Us[0]
    x2, y2 = Us[1]
    x3, y3 = Us[2]
    x4, y4 = Us[3]
    u1, v1 = Vs[0]
    u2, v2 = Vs[1]
    u3, v3 = Vs[2]
    u4, v4 = Vs[3]

    matrixA = np.matrix([[x1,y1,1,0,0,0,-x1*u1,-y1*u1,-u1],[0,0,0,x1,y1,1,-x1*v1,-y1*v1,-v1],
                   [x2,y2,1,0,0,0,-x2*u2,-y2*u2,-u2],[0,0,0,x2,y2,1,-x2*v2,-y2*v2,-v2],
                   [x3,y3,1,0,0,0,-x3*u3,-y3*u3,-u3],[0,0,0,x3,y3,1,-x3*v3,-y3*v3,-v3],
                   [x4,y4,1,0,0,0,-x4*u4,-y4*u4,-u4],[0,0,0,x4,y4,1,-x4*v4,-y4*v4,-v4]])
    
    # SVD  composition
    u, s, v = np.linalg.svd(matrixA)
    # reshape the rightmost singular vector into a 3x3
    H = np.reshape(v[8], (3, 3))

    #normalize and now we have H
    H = (1/H.item(8)) * H
    return H

# normal warp function

def warp_and_augment(im_logo, im_dst, H):
    
    #Given logo image, destination image, and the homography
    #Find the warped final output
    
    imw, imh = im_dst.shape[1], im_dst.shape[0]
    im_warped = cv2.warpPerspective(im_logo, H, (imw, imh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    # get mask for augmented image
    mask = np.array(np.nonzero(im_warped))
    im_out = im_dst.copy()
    for n in range(mask.shape[1]):
        i, j, k = mask[0, n], mask[1, n], mask[2, n]
        im_out[i, j, k] = im_warped[i, j, k]
    return im_warped, im_out
    

# advanced warp function (Bonus)

def warp_and_augment_advanced(im_logo, im_dst, H):
    
    imw, imh = im_dst.shape[1], im_dst.shape[0]
    im_warped = cv2.warpPerspective(im_logo, H, (imw, imh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    # mask creation
    mask_adv = np.zeros_like(im_warped)
    mask_adv[im_warped != 0] = 255
    im_warped_adv = cv2.bitwise_and(im_warped,mask_adv)

    # destination mask
    mask_adv = np.zeros_like(im_dst)
    mask_adv[im_dst != 0] = 255
    im_dst_masked_adv = cv2.bitwise_and(im_dst, mask_adv)

    # final destination image
    im_out = cv2.addWeighted(im_dst_masked_adv, 1, im_warped_adv, 0.5, 0)

    return im_warped_adv, im_out
    
