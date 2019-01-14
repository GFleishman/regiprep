#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# --------------------------- Imports ----------------------------------------------
import sys
from os.path import abspath, isdir, basename, splitext
from os import mkdir, getcwd
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import fileio
import preprocessing as pp


# --------------------------- Interface Specs --------------------------------------
desc = """
~~~***~~**~*   RegiPrep   *~**~~***~~~
Preprocess a pair of confocal images for registration.

Normalizes voxel size, finds a foreground (brain) mask,
and normalizes the field of view. Maintains accurate
values for voxel size, dimensions, and offsets in the
image meta data.

Alternatively, this program can apply the preprocessing
(resampling and cropping/padding) that was already applied
to a reference image to an arbitrary set of new images.
This functionality is turned on with the --transfer_preproc
flag, in which case image1 is the reference and image2
is the image(s) to be resampled/cropped/padded. All
meta data is correctly maintained. The reference image
must have been preprocessed with this program.

This program only accepts images in NIFTI format, with
extension '.nii' or '.nii.gz'. The voxel size must be
correctly set in the meta data and it is assumed that
voxel 0 is the origin, i.e. x,y,z at i,j,k = 0 are all 0.
"""

mandatory_arglist  = ['image1', 'image2']
mandatory_helplist = ["Filepaths to first image channels",
                      "Filepaths to second image channels"]
optional_arglist   = ['--outdir', '--min_vox_size', '--pad_size',
                      '--im1_lambda', '--im2_lambda',
                      '--im1_mask', '--im2_mask',
                      '--transfer_preproc']
optional_helplist  = ['Folder to write outputs to, defaults to where program was executed',
                     'minimum voxel size',
                     'background pad added to each end of each dimension, single int',
                     'lambda param for foreground segmentation of image1',
                     'lambda param for foreground segmentation of image2',
                     'Filepath to mask for image1, prevents computation of new mask',
                     'Filepath to mask for image2, prevents computation of new mask',
                     '1: Turns on transfer mode; any other value runs preprocess mode']


# --------------------------- Functions --------------------------------------------
def parse_inputs():
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    arglist_all = mandatory_arglist + optional_arglist
    helplist_all = mandatory_helplist + optional_helplist
    [parser.add_argument(a, help=b) for a, b in zip(arglist_all, helplist_all)]
    return parser.parse_args()


def read_all_channels(filepaths_string):
    # expected syntax: [[im1_ch1.nii.gz,im1_ch2.nii,...]... ], see mandatory_helplist
    im_paths = [abspath(x.strip('[]')) for x in filepaths_string.split(',')]
    im, im_meta, im_names = [], [], []
    for path in im_paths:
        if not (path[-4:] == '.nii' or path[-7:] == '.nii.gz'):
            print('ERROR: All images must be nifti format ending with .nii or .nii.gz')
            sys.exit()
        data, meta = fileio.read_image(path)
        name = basename(path)[:-7] if path[-3:] == '.gz' else basename(path)[:-4]
        im.append(data); im_meta.append(meta); im_names.append(name)
    return im, im_meta, im_names


def brain_detection(im_list, vox, im_lambda):
    print("\t\tNORMALIZING INTENSITIES")
    """This norm is only for preprocessing purposes, don't overwrite im lists"""
    # TODO: normalize_intensity has 'sigmoid' mode, experiment with that
    im_list_norm = [pp.normalize_intensity(im) for im in im_list]
    # TODO: consider better options here (sum projection in foreground, mean
    #       projection in background - using a quick heuristic foreground/
    #       background threshold)
    #       Use the very nice two-channel image I stitched for Yu/Sujatha
    #       to test alternatives. Do a dual channel alignment to the Z-brain.
    print("\t\tSUMMING CHANNELS")
    im_sum = np.sum(np.array(im_list_norm), axis=0)
    print("\t\tFOREGROUND SEGMENTATION")
    return pp.brain_detection(im_sum, vox, foreground_mask=None,
                              iterations=[60, 25, 5], lambda2=im_lambda)


def slc_positive(x):
    ans = x if x > 0 else None
    return ans

def slc_negative(x):
    ans = x if x < 0 else None
    return ans

def pad_positive(x):
    ans = x if x > 0 else 0
    return ans

def pad_negative(x):
    ans = x if x < 0 else 0
    return ans

# --------------------------- Main -------------------------------------------------
def preprocess(args, outdir):

    print("BEGIN PREPROCESSING")
    print("\tREADING IMAGES")
    """Also initializing some helpful variables for later"""
    im1_list, im1_meta_list, im1_names = read_all_channels(args.image1)
    im2_list, im2_meta_list, im2_names = read_all_channels(args.image2)
    dim = len(im1_list[0].shape)  # image dimension
    v1 = im1_meta_list[0]['pixdim'][1:dim+1]  # im1 voxel size
    v2 = im2_meta_list[0]['pixdim'][1:dim+1]
    im1_brain_mask, im2_brain_mask = None, None
    if args.im1_mask is not None:
        im1_brain_mask, mask_meta = fileio.read_image(args.im1_mask)
    if args.im2_mask is not None:
        im2_brain_mask, mask_meta = fileio.read_image(args.im2_mask)


    print("\tNORMALIZING VOXEL SIZE")
    """All outputs will be at this voxel size, so we can overwrite inputs here
       Update meta data voxel size, important to keep track of all spatial changes"""
    min_size = np.array([0.25,]*dim)
    if args.min_vox_size:
        min_size = np.array([float(x) for x in args.min_vox_size.split('x')])
    for i, (im1, im2) in enumerate(zip(im1_list, im2_list)):
        a, b, newvox = pp.normalize_voxelsize(im1, v1, im2, v2, min_size=min_size)
        im1_list[i], im2_list[i], = a, b
    if im1_brain_mask is not None:
        im1_brain_mask, b, nv = pp.normalize_voxelsize(im1_brain_mask, v1,
                                                       im1_list[0], newvox,
                                                       min_size=min_size)
    if im2_brain_mask is not None:
        im2_brain_mask, b, nv = pp.normalize_voxelsize(im2_brain_mask, v2,
                                                       im2_list[0], newvox,
                                                       min_size=min_size)


    print("\tBRAIN DETECTION")
    if args.im1_mask is None:
        im1_lambda = 20 if args.im1_lambda is None else float(args.im1_lambda)
        im1_brain_mask = brain_detection(im1_list, v1, im1_lambda)
    if args.im2_mask is None:
        im2_lambda = 20 if args.im2_lambda is None else float(args.im2_lambda)
        im2_brain_mask = brain_detection(im2_list, v2, im2_lambda)


    print("\tNORMALIZING FIELD OF VIEW")
    """Padding is after brain detection, so zeros can't affect masking
       Masks will be used for registration, and pads are not within mask
       so pads shouldn't affect registration either"""
    pad_size = 5 if args.pad_size is None else int(args.pad_size)
    im1_box = pp.minimal_bounding_box(im1_brain_mask)
    im2_box = pp.minimal_bounding_box(im2_brain_mask)
    im1_box_dims = np.array([a.stop - a.start for a in im1_box])
    im2_box_dims = np.array([a.stop - a.start for a in im2_box])
    final_dims = np.maximum(im1_box_dims, im2_box_dims) + 2*pad_size
    diff1, diff2 = (final_dims - im1_box_dims)/2., (final_dims - im2_box_dims)/2.
    pad1 = np.array([(np.ceil(d), np.floor(d)) for d in diff1]).astype(int)
    pad2 = np.array([(np.ceil(d), np.floor(d)) for d in diff2]).astype(int)
    for i, (im1, im2) in enumerate(zip(im1_list, im2_list)):
        im1_list[i] = np.pad(im1[im1_box], pad1, mode='constant')
        im2_list[i] = np.pad(im2[im2_box], pad2, mode='constant')
    im1_brain_mask = np.pad(im1_brain_mask[im1_box], pad1, mode='constant')
    im2_brain_mask = np.pad(im2_brain_mask[im2_box], pad2, mode='constant')


    print("\tUPDATING METADATA")
    """Preprocessing changes spatial relationship with raw image,
       which will be recorded in sform; ITK uses qform, and this
       which must be zeros for registration to execute accurately"""
    xyz = ['x', 'y', 'z']
    im1_origin = np.array([a.start - np.ceil(b) for a, b in zip(im1_box, diff1)])*newvox
    im2_origin = np.array([a.start - np.ceil(b) for a, b in zip(im2_box, diff2)])*newvox
    for meta1, meta2 in zip(im1_meta_list, im2_meta_list):
        meta1['sform_code'], meta1['qform_code'] = 2, 1
        meta2['sform_code'], meta2['qform_code'] = 2, 1
        meta1['pixdim'][1:dim+1], meta1['dim'][1:dim+1] = newvox, final_dims
        meta2['pixdim'][1:dim+1], meta2['dim'][1:dim+1] = newvox, final_dims
        for i, (a, b, c) in enumerate(zip(xyz, im1_origin, im2_origin)):
            meta1['srow_'+a][i], meta2['srow_'+a][i] = newvox[i], newvox[i]
            meta1['srow_'+a][-1], meta2['srow_'+a][-1] = b, c
            meta1['qoffset_'+a], meta2['qoffset_'+a] = 0, 0
        if all([meta1['quatern_'+a] == 0 for a in ['b', 'c', 'd']]):
            meta1['quatern_d'] = 1
        if all([meta2['quatern_'+a] == 0 for a in ['b', 'c', 'd']]):
            meta2['quatern_d'] = 1


    print("\tWRITING IMAGES")
    fileio.write_image(outdir+'/'+im1_names[0]+'_mask.nii.gz', im1_brain_mask, im1_meta_list[0])
    fileio.write_image(outdir+'/'+im2_names[0]+'_mask.nii.gz', im2_brain_mask, im2_meta_list[0])
    for i in range(len(im1_list)):
        fileio.write_image(outdir+'/'+im1_names[i]+'_pp.nii.gz', im1_list[i], im1_meta_list[i])
        fileio.write_image(outdir+'/'+im2_names[i]+'_pp.nii.gz', im2_list[i], im2_meta_list[i])
    print("PREPROCESSING COMPLETE")




def transfer_preprocess(args, outdir):

    print("BEGIN PREPROCESSING TRANSFER")
    print("\tREADING IMAGES")
    im1_list, im1_meta_list, im1_names = read_all_channels(args.image1)
    im2_list, im2_meta_list, im2_names = read_all_channels(args.image2)
    if not len(im1_list) == 1:
        print("ERROR: Only one reference allowed in transfer mode")
        sys.exit()

    print("\tGETTING PREPROCESSING PARAMETERS")
    """assumed that all images in im2_list have the same shape, origin, and voxel size"""
    d = len(im1_list[0].shape)  # image dimensionality
    origin = np.array([im1_meta_list[0]['srow_'+a][-1] for a in ['x', 'y', 'z']])
    v1, v2 = im1_meta_list[0]['pixdim'][1:d+1], im2_meta_list[0]['pixdim'][1:d+1]
    target_dims = np.array(im1_list[0].shape)
    current_dims = np.round(np.array(im2_list[0].shape) * v2/v1).astype(np.int)
    left_offsets = (origin/v1).astype(int)
    right_offsets = target_dims - current_dims + left_offsets
    box = [slice(slc_positive(l), slc_negative(r)) for l, r in zip(left_offsets, right_offsets)]
    pad = [(-pad_negative(l), pad_positive(r)) for l, r in zip(left_offsets, right_offsets)]

    print("\tAPPLYING PREPROCESSING")
    for im, name in zip(im2_list, im2_names):
        im_pp = pp._resample(im, vox=(v2, v1))
        im_pp = np.pad(im_pp[box], pad, mode='constant')
        fileio.write_image(outdir+'/'+name+'_pp.nii.gz', im_pp, im1_meta_list[0])
    print("\tPREPROCESSING TRANSFER COMPLETE")



# TODO: implement a third mode: revert_preprocess (undo preprocessing relative to reference)
#       alternative: rewrite transfer_preprocess to consider origin of im2, compute offsets
#       from both origins, then "revert" would just be "transfer_preprocess" using the
#       original image as the reference
# TODO: add a "reformat" mode that takes arbitrary image formats as input and writes out
#       as .nii.gz
# TODO: add a metadata updater mode
# TODO: generally the goal with the above two is to eliminate c3d from the processing pipeline
if __name__ == '__main__':

    print("PARSING INPUTS")
    args = parse_inputs()
    outdir = args.outdir if args.outdir is not None else abspath(getcwd())
    if not isdir(outdir): mkdir(outdir)

    print("DETERMINING MODE")
    if args.transfer_preproc == '1':
        transfer_preprocess(args, outdir)
    else:
        preprocess(args, outdir)

