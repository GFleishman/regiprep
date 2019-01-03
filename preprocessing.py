#!/usr/bin/env python3
# -*- coding: utf-8 -*- 


# --------------------------- Imports ----------------------------------------------
import sys
import numpy as np
import scipy.ndimage as ndi
import scipy.ndimage.filters as ndif
import scipy.ndimage.morphology as ndim
import scipy.ndimage.measurements as ndims
import morphsnakes


# --------------------------- Internal Functions -----------------------------------
def _background_statistics(image, rad=5):
    """Find threshold which approximately separates foreground from background
       Assumes image has not been padded, i.e. background noise extends to corners"""
    a, b = slice(0, rad), slice(-rad, None)
    corners = [[a,a,a], [a,a,b], [a,b,a], [a,b,b],
               [b,a,a], [b,a,b], [b,b,a], [b,b,b]]
    mean = np.median([np.mean(image[c]) for c in corners])
    std = np.median([np.std(image[c]) for c in corners])
    return mean, std


def _resample(img, vox=(), res=[], order=1):
    """Resample image to voxel size or specified resolution"""
    if (not (vox or res)) or (vox and res):
        print("Must specify voxel or resolution, but not both")
        sys.exit()
    elif vox:
        vox = [np.array(v).astype(float) for v in vox]
        factor = vox[0]/vox[1]
    elif res:
        factor = np.array(res).astype(float)/np.array(img.shape).astype(float)
    if np.allclose(factor, np.ones(len(factor))):
        return img
    else:
        return ndi.zoom(img, factor, order=order)


def _segment(image, lambda2, iterations, smoothing=1,
             sigma=None, threshold=None, init=None):
    """Segment a region of the image using morphological active contours""" 
    if threshold is not None:
        image[image < threshold] = 0
    if sigma is not None:
        image = ndif.gaussian_filter(image, sigma=sigma)
    if init is None:
        init = np.zeros_like(image, dtype=np.uint8)
        bounds = np.ceil(np.array(image.shape) * 0.1).astype(int)
        init[[slice(b, -b) for b in bounds]] = 1
    macwe = morphsnakes.MorphACWE(image, smoothing=smoothing, lambda2=lambda2)
    macwe.levelset = init
    macwe.run(iterations)
    return macwe.levelset.astype(np.uint8)


def _largest_connected_component(mask):
    lbls, nlbls = ndims.label(mask)
    vols = ndims.labeled_comprehension(mask, lbls, range(1, nlbls+1), np.sum, float, 0)
    mask[lbls != np.argmax(vols)+1] = 0
    return mask


# --------------------------- External Functions -----------------------------------
# TODO: add functions for functional calcium imaging preprocessing

def normalize_intensity(image, mode='linear', **kwargs):
    """Remapping of intensity range using either linear or sigmoid function.
       Intensities above clip percentile are truncated to clip percentile."""
    perc = kwargs['clip'] if 'clip' in kwargs.keys() else 99.5
    clip, perc100 = np.percentile(image, [perc, 100])
    if clip != perc100: image[image > clip] = clip
    newmin = kwargs['min'] if 'min' in kwargs.keys() else 0
    newmax = kwargs['max'] if 'max' in kwargs.keys() else 255
    newmin, newmax = np.float32(newmin), np.float32(newmax)
    if mode.lower() == 'linear':
        slope = (newmax - newmin)/(image.max() - image.min())
        return (image - image.min()) * slope + newmin
    elif mode.lower() == 'sigmoid':
        a = kwargs['alpha'] if 'alpha' in kwargs.keys() else 2.5*np.std(image)
        b = kwargs['beta'] if 'beta' in kwargs.keys() else (image.max()-image.min())/2.
        return (newmax - newmin) * 1./(1. + np.exp(-(image - b)/a)) + newmin


# TODO: this is the weakest method, can it be improved?
def foreground_detection(image, voxel, subsample=4, iterations=50, lambda2=100):
    """Create mask of foreground voxels"""
    new_res = np.ceil(np.array(image.shape) / subsample).astype(int)
    image_small = _resample(image, res=list(new_res))
    mean, std = _background_statistics(image)
    mask = _segment(image_small, lambda2=lambda2, iterations=iterations,
                    sigma=4./voxel, threshold=mean+std)
    # fish shape does not have holes along any of the orthogonal cross section
    for axis in range(len(mask.shape)):
        slc = [slice(None)]*len(mask.shape)
        for i in range(mask.shape[axis]):
            slc[axis] = slice(i, i+1)
            m = ndim.binary_fill_holes(mask[slc].squeeze())
            mask[slc] = np.expand_dims(m, axis)
    return _resample(mask, res=image.shape, order=0)


# TODO: change default for foreground_mask to None
def brain_detection(image, voxel, foreground_mask,
                    subsample=[4, 2, 1], iterations=[40, 8, 2], lambda2=20):
    """Create mask of brain region only"""
    if foreground_mask is not None:
        se = np.ones((5,)*len(foreground_mask.shape))
        mask = ndim.binary_erosion(foreground_mask, structure=se, iterations=3)
        mask = _largest_connected_component(mask)
    else:
        mask = None
    mean, std = _background_statistics(image)
    for i, s in enumerate(subsample):
        new_res = np.ceil(np.array(image.shape) / s).astype(int)
        if tuple(new_res) == image.shape:
            image_small = image
        else:
            image_small = _resample(image, res=list(new_res))
        if mask is not None:
            mask = _resample(mask, res=list(new_res), order=0)
        mask = _segment(image_small, lambda2=lambda2, iterations=iterations[i],
                        sigma=8./voxel, threshold=mean, init=mask)
    # simple topological corrections
    mask = ndim.binary_erosion(mask, iterations=2)
    mask = _largest_connected_component(mask)
    mask = ndim.binary_dilation(mask, iterations=2)
    mask = ndim.binary_fill_holes(mask).astype(np.uint8)
    if mask.shape != image.shape: mask = _resample(mask, res=image.shape, order=0)
    return mask


def normalize_voxelsize(image1, vox1, image2, vox2, min_size=None):
    """Resamples both images to minimum voxel size"""
    minimum_voxel_size = [min([vox1[i], vox2[i]]) for i in range(len(vox1))]
    if min_size is not None:
        minimum_voxel_size = [min_size[i] if min_size[i] > minimum_voxel_size[i] else
                              minimum_voxel_size[i] for i in range(len(minimum_voxel_size))]
    image1_new = _resample(image1, vox=(vox1, minimum_voxel_size))
    image2_new = _resample(image2, vox=(vox2, minimum_voxel_size))
    return image1_new, image2_new, minimum_voxel_size


def minimal_bounding_box(labels):
    labels = np.array(labels).astype(int)
    labels[labels > 0] = 1
    return ndims.find_objects(labels, max_label=1)[0]


def normalize_extent(image1, image2):
    """Pad both images to maximum voxel grid size"""
    sh1, sh2 = image1.shape, image2.shape
    maxsh = np.array([max([sh1[i], sh2[i]]) for i in range(len(sh1))])
    diff1, diff2 = (maxsh-sh1)/2., (maxsh-sh2)/2.
    pad1 = [(np.ceil(diff1[i]), np.floor(diff1[i])) for i in range(len(sh1))]
    pad2 = [(np.ceil(diff2[i]), np.floor(diff2[i])) for i in range(len(sh2))]
    image1_new = np.pad(image1, np.array(pad1).astype(int), mode='constant')
    image2_new = np.pad(image2, np.array(pad2).astype(int), mode='constant')
    return image1_new, image2_new


def resample(img, vox=(), res=[], order=1):
    return _resample(img, vox, res, order)

