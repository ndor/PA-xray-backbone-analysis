import numpy as np
import cv2
import h5py
import os
import itertools as it
from keras.layers import Input, concatenate, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


def resize_to(image, minimal_size_p):
    im = image.copy()
    h, w = im.shape
    f = min(h, w)
    if f > minimal_size_p:
        scale = 1. * f / minimal_size_p
        h_ = int(h / scale)
        w_ = int(w / scale)
        im = cv2.resize(im, (w_, h_))
    return im


def bbox(img):
    img = (img > 0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return rmin, rmax, cmin, cmax


def crop_xray(image, mask=None):
    im = image.copy()
    h, w = im.shape
    def toCropByUnique(uniques, counts):
        to_crop = False
        counts_threshold = int((h * w) ** 0.25)
        if len(uniques) <= counts_threshold:
            to_crop = True
        if not to_crop:
            counts_span_threshold = int(0.95 * max(h, w))
            if counts.max() >= counts_span_threshold:
                to_crop = True
        return to_crop
    # unique values count:
    # X:
    mask_x = np.zeros_like(im)
    for i in range(w):
        uniques, counts = np.unique(im[:, i], return_counts=True)
        if not toCropByUnique(uniques, counts):
            mask_x[:, i] = 255
    # Y:
    mask_y = np.zeros_like(im)
    for i in range(h):
        uniques, counts = np.unique(im[i, :], return_counts=True)
        if not toCropByUnique(uniques, counts):
            mask_y[i, :] = 255
    mask_unique = mask_x / 2 + mask_y / 2 + 1
    mask_unique[mask_unique < 255] = 0
    im[mask_unique == 0] = 0
    rmin, rmax, cmin, cmax = bbox(im)
    im = im[rmin: rmax, cmin: cmax]
    if mask is not None:
        mask = mask[rmin: rmax, cmin: cmax]
        return im, mask
    return im


def bilateral_filter(image):
    im = image.copy()
    h, w = im.shape
    n = min(h, w) / 9
    if n % 2 != 1: n += 1
    d = 3
    sigmaColor = np.mean(im)
    sigmaSpace = n
    im = cv2.bilateralFilter(im, d, sigmaColor, sigmaSpace)
    return im


def denoise(image):
    im = image.copy()
    # kernel = np.ones((3, 3), dtype='uint8')
    # it = min(im.shape) / 1024
    # image = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=it)
    cv2.fastNlMeansDenoising(im, im, h=3., templateWindowSize=7, searchWindowSize=21)
    return image


def sharpen(image):
    im = image.copy()
    h, w = im.shape
    # sharpening the image:
    # Create the identity filter, but with the 1 shifted to the right!
    k = min(h, w) / 9
    if k % 2 ==0: k += 1
    kernel = np.zeros((k, k), dtype='float32')
    kernel[(k - 1) / 2, (k - 1) / 2] = 2.0  # Identity, times two!
    # Create a box filter:
    boxFilter = np.ones((k, k), dtype='float32') / (1.*k**2)
    # Subtract the two:
    kernel = kernel - boxFilter
    im = cv2.filter2D(im, -1, kernel)
    cv2.fastNlMeansDenoising(im, im, h=3., templateWindowSize=7, searchWindowSize=21)
    return im


def normalize(image):
    im = image.copy()
    im = cv2.equalizeHist(im)
    return im


def local_norm(image):
    im = image.copy()
    h, w = im.shape
    k = min(h, w) / 9
    if k % 2 != 1: k += 1
    float_im = im.astype('float32') / 255.0
    def local_normalization(float_im, k):
        blur = cv2.GaussianBlur(float_im, (0, 0), sigmaX=k, sigmaY=k)
        num = float_im - blur
        blur = cv2.GaussianBlur(num * num, (0, 0), sigmaX=k, sigmaY=k)
        den = cv2.pow(blur + 1e-7, 0.5)
        im = num / den
        # cv2.normalize(im, dst=im, alpha=0.0, beta=255., norm_type=cv2.NORM_MINMAX).astype('uint8')
        return im

    # mink = 2#k / 2
    # for ki in range(mink, k + 1, 1):
    #     if ki == mink:
    #         nlayer = local_normalization(float_im.copy(), ki) * ki
    #     else:
    #         nlayer = nlayer + local_normalization(float_im.copy(), ki) * ki

    nlayer = local_normalization(float_im.copy(), k / 2)

    # nlayer = nlayer / (k - 3)
    # nlayer += abs(nlayer.min())
    # im = normalize(im)
    # nlayer = nlayer * im
    # nlayer = local_normalization(nlayer, ki) / ki
    cv2.normalize(nlayer, dst=nlayer, alpha=0.0, beta=255., norm_type=cv2.NORM_MINMAX)
    nlayer = nlayer.astype('uint8')
    return nlayer


def BHThreshold(image):
    im = image.copy()
    hist, bins = np.histogram(im, 256, [0, 256])
    WL = 0
    WR = 1
    i = 1
    while WL <= WR:
        WL = (hist[:i] * np.arange(len(hist[:i]), 0, -1)).sum()
        WR = (hist[i:] * np.arange(1, len(hist[i:]) + 1, 1)).sum()
        i += 1
    ret, mask = cv2.threshold(im, i, 255, cv2.THRESH_BINARY)
    # cv2.imshow('p', mask)
    # cv2.waitKey(0)
    return mask


def adaptive_threshold(image):
    im = image.copy()
    h, w = im.shape
    S = w/8
    s2 = S/2
    T = 15.0
    #integral img
    int_img = cv2.integral(im)
    #output img
    mask = np.zeros_like(im)
    for col in range(w):
        for row in range(h):
            #SxS region
            y0 = max(row-s2, 0)
            y1 = min(row+s2, h-1)
            x0 = max(col-s2, 0)
            x1 = min(col+s2, w-1)
            count = (y1-y0)*(x1-x0)
            sum_ = int_img[y1, x1]-int_img[y0, x1]-int_img[y1, x0]+int_img[y0, x0]
            if im[row, col] * count < sum_ * (100. - T) / 100.:
                mask[row,col] = 0
            else:
                mask[row,col] = 255
    return mask


def otsu_thresh(image):
    im = image.copy()
    threshold, otsu_mask = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu_mask, threshold


def gauss_thresh(image):
    im = image.copy()
    h, w = im.shape
    n = min(h, w) / 9
    if n % 2 != 1: n += 1
    C = 0
    gauss_mask = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, n, C)
    return gauss_mask


def mean_thresh(image):
    im = image.copy()
    h, w = im.shape
    n = min(h, w) / 9
    if n % 2 != 1: n += 1
    C = 0
    mean_mask = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, n, C)

    return mean_mask


def MSER(image):
    im = image.copy()
    h, w = im.shape
    mser = cv2.MSER()
    regions = mser.detect(im, None)
    MSER_mask = np.zeros_like(im)
    for r in regions:
        MSER_mask[r[:, 1], r[:, 0]] = 255
    return MSER_mask


def auto_canny(image, sigma=0.33):
    im = image.copy()
    h, w = im.shape
    v = np.median(im)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(im, lower, upper)
    return edged


def off_body_bits_cleaner(body_mask):
    mask = body_mask.copy()
    h, w = mask.shape
    im_size = h * w
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        m = np.zeros_like(mask)
        cv2.drawContours(m, cnts, i, 255, -1)
        if np.count_nonzero(m) < im_size / 4:
            mask[m == mask] = 0
    return mask


def in_body_bits_cleaner(body_mask):
    mask = body_mask.copy()
    h, w = mask.shape
    im_size = h * w
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(cnts)):
        m = np.zeros_like(mask)
        cv2.drawContours(m, cnts, i, 255, -1)
        if np.count_nonzero(m) < im_size / 100:
            mask[m > mask] = 255
    return mask


def skeletonize(binary_mask):
    size = np.size(binary_mask)
    skel = np.zeros_like(binary_mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while (not done):
        eroded = cv2.erode(binary_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_mask = eroded.copy()
        zeros = size - cv2.countNonZero(binary_mask)
        if zeros == size:
            done = True
    return skel


def parse_body(image):
    im = image.copy()
    im = normalize(im)
    im[im>255/3] = 255/3

    d_ = 32
    im = resize_to(im, d_)
    h, w = im.shape

    mask0, threshold = otsu_thresh(im)

    mask1 = np.zeros_like(im)
    for i in range(h):
        mask, threshold = otsu_thresh(im[i, :])
        mask1[i, :] = mask.flatten()

    mask2 = np.zeros_like(im)
    for i in range(w):
        mask, threshold = otsu_thresh(im[:, i])
        mask2[:, i] = mask.flatten()

    mask3 = BHThreshold(im)
    mask4 = adaptive_threshold(im)
    mask_ = mask0 | mask1 | mask2 | mask3 | mask4
    mask = np.zeros_like(im, dtype='uint8')
    mask[mask_ > 0] = 1


    rect = (0, 0, im.shape[1] - 1, im.shape[0] - 1)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(np.dstack([im, im, im]), mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    im = image.copy()
    h, w = im.shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    mean_bg = np.mean(im[mask==0])
    ret, body_mask = cv2.threshold(im, mean_bg, 255, cv2.THRESH_BINARY)

    # cleaning up:
    body_mask = off_body_bits_cleaner(body_mask)
    body_mask = in_body_bits_cleaner(body_mask)
    kernel = np.ones((5, 5), dtype='uint8')
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
    return body_mask


def standardize(image):
    im = image.copy()
    im = crop_xray(im)
    # cv2.imshow('p', im)
    # cv2.waitKey(0)
    im = denoise(im)
    im = bilateral_filter(im)
    body_mask = parse_body(im)
    # cv2.imshow('p', body_mask)
    # cv2.waitKey(0)
    # remove background:
    im -= im[body_mask == 0].min()
    im[body_mask == 0] = 0
    # cv2.imshow('p', im)
    # cv2.waitKey(0)
    im, body_mask = crop_xray(im, body_mask)
    # cv2.imshow('p', im)
    # cv2.waitKey(0)
    return im, body_mask


def make_input_image(image_path, result_size=(196, 128)):
    im = cv2.imread(image_path, 0)
    im, body_mask = standardize(im)
    h, w = im.shape
    zeros = np.zeros_like(im)

    im = sharpen(im)
    im = normalize(im)
    im_norm = local_norm(im)
    im_norm = sharpen(im_norm)
    im_norm = normalize(im_norm)

    im[body_mask == 0] = 0
    im_norm[body_mask == 0] = 0
    final_stack = np.dstack([zeros, im_norm, im])
    if result_size is not None:
        fy = 1. * result_size[0] / h
        fx = 1. * result_size[1] / w
        k = min(fx, fy)
        final_stack = cv2.resize(final_stack, None, dst=None, fx=k, fy=k)
        h_, w_, d_ = final_stack.shape
        y_pad = 0
        x_pad = 0
        if h_ < result_size[0]:
            y_pad = (result_size[0] - h_) / 2
        if w_ < result_size[1]:
            x_pad = (result_size[1] - w_) / 2

        y_pad2 = y_pad
        x_pad2 = x_pad
        if 2 * y_pad < result_size[0] - h_:
            y_pad2 += 1
        if 2 * x_pad < result_size[1] - w_:
            x_pad2 += 1

        final_stack = cv2.copyMakeBorder(final_stack, y_pad, y_pad2, x_pad, x_pad2, cv2.BORDER_CONSTANT, value=0)

    x1 = final_stack[:, :, 1]
    x2 = final_stack[:, :, 2]
    Lx1 = cv2.Laplacian(x1, cv2.CV_8U)
    Lx2 = cv2.Laplacian(x2, cv2.CV_8U)
    X = np.dstack([x1, Lx1, x2, Lx2]).astype('float32') / 255
    # X = np.array(X, dtype='float32')
    return X


# def net(size=(196, 128, 4)):
#     model_description = 'PAnet'
#     act = 'relu'
#     pool_size = 2
#     k_num = 32
#     input = Input(shape=size)
#
#
#     C1 = []
#     for d in range(1, 4, 1):
#         C1.append(Conv2D(k_num, 5, strides=(1, 1), padding='same', dilation_rate=(d, d), activation=act)(input))
#         # C1.append(Conv2D(k_num, 7, strides=(1, 1), padding='same', dilation_rate=(d, d), activation=act)(input))
#     C1 = concatenate(C1)
#     MP1 = MaxPooling2D(pool_size=(pool_size, pool_size))(C1)
#     AP1 = AveragePooling2D(pool_size=(pool_size, pool_size))(C1)
#     P1 = concatenate([MP1, AP1])
#
#
#     C2 = []
#     for d in range(1, 3, 1):
#         C2.append(Conv2D(2 * k_num, 3, strides=(1, 1), padding='same', dilation_rate=(d, d), activation=act)(P1))
#         # C2.append(Conv2D(2 * k_num, 5, strides=(1, 1), padding='same', dilation_rate=(d, d), activation=act)(P1))
#     C2 = concatenate(C2)
#     MP2 = MaxPooling2D(pool_size=(pool_size, pool_size))(C2)
#     AP2 = AveragePooling2D(pool_size=(pool_size, pool_size))(C2)
#     P2 = concatenate([MP2, AP2])
#
#
#     C3 = Conv2D(4 * k_num, 3, strides=(1, 1), padding='same', dilation_rate=(1, 1), activation=act)(P2)
#     U1 = UpSampling2D((pool_size, pool_size))(C3)
#     C3 = concatenate([P1, C2, U1])
#
#
#     C4 = []
#     for d in range(1, 3, 1):
#         C4.append(Conv2D(2 * k_num, 3, strides=(1, 1), padding='same', dilation_rate=(d, d), activation=act)(C3))
#         # C4.append(Conv2D(2 * k_num, 5, strides=(1, 1), padding='same', dilation_rate=(d, d), activation=act)(C3))
#     C4 = concatenate(C4)
#     U2 = UpSampling2D((pool_size, pool_size))(C4)
#     C3 = concatenate([input, C1, U2])
#
#
#     mask = Conv2D(1, 3, strides=(1, 1), padding='same', dilation_rate=(1, 1), activation='sigmoid')(C3)#C4
#
#
#     model = Model(inputs=input, outputs=mask)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
#     ############################################################################################
#     # if previous file exist:
#     if os.path.isfile(model_description + '.hdf5'):
#         print 'loading weights file: ' + os.path.join(model_description + '.hdf5')
#         model.load_weights(model_description + '.hdf5')
#     ############################################################################################
#     return model


def net(size=(196, 128, 4)):
    model_description = 'printXray'
    act = 'relu'
    input = Input(shape=size)

    C1 = Conv2D(32, 5, padding='same', activation=act)(input)
    MP1 = MaxPooling2D(pool_size=(2, 2))(C1)
    C2 = Conv2D(64, 3, padding='same', activation=act)(MP1)
    MP2 = MaxPooling2D(pool_size=(2, 2))(C2)
    C3 = Conv2D(128, 3, padding='same', activation=act)(MP2)
    MP3 = MaxPooling2D(pool_size=(2, 2))(C3)
    C4 = Conv2D(256, 3, padding='same', activation=act)(MP3)
    U1 = UpSampling2D((2, 2))(C4)
    # U1 = concatenate([U1, C3])
    T1 = Conv2DTranspose(128, 3, padding='same', activation=act)(U1)
    U2 = UpSampling2D((2, 2))(T1)
    U2 = ZeroPadding2D(padding=(1, 0))(U2)
    U2 = concatenate([U2, C2])
    T3 = Conv2DTranspose(64, 3, padding='same', activation=act)(U2)
    U3 = UpSampling2D((2, 2))(T3)
    U3 = concatenate([U3, C1])
    T3 = Conv2DTranspose(32, 3, padding='same', activation=act)(U3)

    U4 = concatenate([T3, input])
    mask = Conv2D(1, 3, padding='same', activation='sigmoid')(U4)

    model = Model(inputs=input, outputs=mask)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    ############################################################################################
    # if previous file exist:
    if os.path.isfile(model_description + '.hdf5'):
        print 'loading weights file: ' + os.path.join(model_description + '.hdf5')
        model.load_weights(model_description + '.hdf5')
    ############################################################################################
    return model


def spine_parser(x):
    x0 = x.copy()
    h, w, d = x.shape
    model = net()
    x = x.reshape(1, h, w, d)
    y = model.predict_on_batch(x)[0, :, :, 0]
    print x.min(), x.max(), x.mean()
    print y.min(), y.max(), y.mean(), y.dtype
    y[y < 0.5] = 0
    y[y > 0] = 1
    y = 255 * y
    y = y.astype('uint8')
    # print y.shape
    # cv2.imshow('p', np.hstack([x0[:, :, 0], x0[:, :, 1], x0[:, :, 2], x0[:, :, 3], y]))
    # cv2.waitKey(0)
    return y


def spine_spline(image, spine_mask):
    h, w = spine_mask.shape
    # distance = cv2.distanceTransform(spine_mask, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_5)
    # # cv2.normalize(distance, distance, 0, 255., cv2.NORM_MINMAX)
    # # distance = distance.astype('uint8')
    # spine_y_top = 0
    # for dy in range(h):
    #     if np.count_nonzero(distance[dy, :]) > 0:
    #         spine_y_top = dy
    #         break

    # assuming mask is solid and single:
    kernel = np.ones((2, 2), np.uint8)
    spine_mask = cv2.morphologyEx(spine_mask, cv2.MORPH_CLOSE, kernel)
    spine_mask = cv2.morphologyEx(spine_mask, cv2.MORPH_OPEN, kernel)
    spine_skel = skeletonize(spine_mask)
    skel_location = np.where(spine_skel>0)
    z = np.polyfit(skel_location[0], skel_location[1], 9)
    p = np.poly1d(z)
    pd1 = np.polyder(p)
    x_ = np.arange(h, dtype='int')
    y_ = np.round(p(x_)).astype('int')
    dy_ = pd1(x_)

    # check where spline should start and end:
    y_[y_ > w - 1] = w - 1
    y_[y_ < 0] = 0
    spine_poly_mask = np.zeros_like(spine_mask)
    spine_poly_mask[x_, y_] = 255
    spine_poly_mask[spine_mask == 0] = 0
    splinie = np.sum(spine_poly_mask, axis=-1, dtype='int')
    cut_ends_for_spline_mid = np.count_nonzero(splinie) * 0.1
    S = np.where(splinie > 0)[0]
    spline_top_index = S.min() + int(cut_ends_for_spline_mid)
    spline_bottom_index = S.max() - int(cut_ends_for_spline_mid)

    # angles:
    angles = np.arctan(dy_) * 180 / np.pi
    normalized_angles = angles - angles[spline_top_index:spline_bottom_index].mean()

    # Cobb angle:
    max_cobb = normalized_angles[spline_top_index:spline_bottom_index].max()
    min_cobb = normalized_angles[spline_top_index:spline_bottom_index].min()
    index_max_cobb = np.where(normalized_angles == max_cobb)
    index_min_cobb = np.where(normalized_angles == min_cobb)
    cobb_angle = max_cobb - min_cobb

    # display:
    spine_mask = spine_mask / 2 + spine_poly_mask / 2
    # cv2.imshow('p', np.hstack([image[:, :, -1], image[:, :, -2], spine_mask, spine_skel]))
    # cv2.waitKey(0)
    im = np.dstack([image[:, :, -1], image[:, :, -1], image[:, :, -1]])
    kernel = np.ones((3, 3), dtype='uint8')
    spine_poly_mask = cv2.dilate(spine_poly_mask, kernel)
    im[:, :, 0][spine_poly_mask > 0] = 0
    im[:, :, 1][spine_poly_mask > 0] = 0
    im[:, :, 2][spine_poly_mask > 0] = 255

    dx_line = w / 3
    invdy = 1. / dy_

    p0 = (y_[index_max_cobb], x_[index_max_cobb])
    cv2.circle(im, p0, 8, (255, 0, 0), thickness=2, lineType=8, shift=0)
    pdy = invdy[index_max_cobb]
    pdx = dy_[index_max_cobb]
    p1 = (p0[0] + 8, p0[1] + int(-pdx * 8))
    p2 = (p0[0] + dx_line, p0[1] + int(-pdx * dx_line))
    cv2.line(im, p1, p2, (255, 0, 0), thickness=2, lineType=8, shift=0)
    p1 = (p0[0] - 8, p0[1] + int(pdx * 8))
    p2 = (p0[0] - dx_line, p0[1] + int(pdx * dx_line))
    cv2.line(im, p1, p2, (255, 0, 0), thickness=2, lineType=8, shift=0)

    p0 = (y_[index_min_cobb], x_[index_min_cobb])
    cv2.circle(im, p0, 8, (255, 0, 0), thickness=2, lineType=8, shift=0)
    pdy = invdy[index_min_cobb]
    pdx = dy_[index_min_cobb]
    p1 = (p0[0] + 8, p0[1] + int(-pdx * 8))
    p2 = (p0[0] + dx_line, p0[1] + int(-pdx * dx_line))
    cv2.line(im, p1, p2, (255, 0, 0), thickness=2, lineType=8, shift=0)
    p1 = (p0[0] - 8, p0[1] + int(pdx * 8))
    p2 = (p0[0] - dx_line, p0[1] + int(pdx * dx_line))
    cv2.line(im, p1, p2, (255, 0, 0), thickness=2, lineType=8, shift=0)

    cobb_txt = ['measured', 'Cobb angle:' ,str(cobb_angle)[:5] + ' deg`']
    font_scale = 2. * w / 900
    font = cv2.FONT_HERSHEY_SIMPLEX
    df = 0
    for tx in cobb_txt:
        org = (8, p1[1] + df)
        cv2.putText(im, tx, org, font, font_scale, (0, 0, 255), 2)
        df += font_scale * w / 8

    # cv2.imshow('p', im)
    # cv2.waitKey(0)


    return im#spine_poly_mask




if __name__ == '__main__':
    # image_path = 'samples/brace pa.jpg'
    # x = make_input_image(image_path)
    # y = spine_parser(x)

    for i in range(21):
        path = '/home/nate/Desktop/oren/basePAdata/PAdataset/' + str(i) + '.png'
        x = cv2.imread(path, 1)
        x = cv2.resize(x, None, fx=0.25, fy=0.25)
        y = x[:, :, 0]
        im = spine_spline(x, y)
        # cv2.imwrite('/home/nate/Desktop/' + str(i) + '.png', im)
        cv2.imshow('p', im)
        cv2.waitKey(0)
        print i





