import cv2
import numpy as np,sys
import scipy.signal

DEPTH = 2

def crop_to_match(target, img_2):
    if target.shape[0] < img_2.shape[0]:
        img_2 = img_2[0:-1,:]
    if target.shape[1] < img_2.shape[1]:
        img_2 = img_2[:,0:-1]
    return img_2

def gaussPyr(image, depth):
    G = image.copy()
    g_pyr = [G]
    for i in xrange(depth):
        # cv2.imshow("gauss", G)
        # cv2.waitKey(500)
        G = cv2.pyrDown(G)
        g_pyr.append(G)
    return g_pyr

def laplPyr(g_pyr):
    l_pyr = [ g_pyr[len(g_pyr)-1] ]
    for i in xrange(len(g_pyr)-1,0,-1):
        GE = cv2.pyrUp(g_pyr[i])
        GE = crop_to_match(g_pyr[i-1], GE)
        L = cv2.subtract(g_pyr[i-1], GE)
        # cv2.imshow("lapl", L)
        # cv2.waitKey(500)
        l_pyr.append(L)
    return l_pyr

def blend_and_collapse(lpB, lpW, gpM):
    LS = []
    for b,w,m in zip(lpB,lpW,gpM[::-1]):
        b_comp = (np.ones(m.shape, dtype=np.float32) - m) * b.astype(np.float32)
        w_comp = m * w.astype(np.float32)
        ls = b_comp.astype(np.uint8) + w_comp.astype(np.uint8)
        # cv2.imshow("blended lapl", ls)
        # cv2.waitKey(500)
        LS.append(ls)

    img = LS[0]
    for i in xrange(1,len(LS)):
        img = cv2.pyrUp(img)
        img = crop_to_match(LS[i], img)
        img = cv2.add(img, LS[i])
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def blend(black, white, mask):
    # generate Gaussian/Laplacian pyramid for black
    gp_black = gaussPyr(black, DEPTH)
    lp_black = laplPyr(gp_black)

    # generate Gaussian/Laplacian pyramid for white
    gp_white = gaussPyr(white, DEPTH)
    lp_white = laplPyr(gp_white)

    # generate Gaussian pyramid for mask
    gp_mask = gaussPyr(mask, DEPTH)

    # now blend reconstruct
    blended = blend_and_collapse(lp_black,lp_white,gp_mask)
    return blended

if __name__ == "__main__":
    # black = cv2.imread('black.jpg')
    # white = cv2.imread('white.jpg')
    # mask = cv2.imread('mask.jpg').astype(np.float32)/255.0

    # img = blend(black,white,mask)
    # cv2.imshow("blend", img)
    # cv2.imwrite("blended.jpg", img)
    cat = cv2.imread('cat.jpg')
    gc = gaussPyr(cat, 4)
    lc = laplPyr(gc)

    cv2.waitKey(0)
    cv2.destroyAllWindows()