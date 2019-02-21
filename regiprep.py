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
Prepare images for deformable registration.
"""

epi = """
RegiPrep is four programs in one. Each call to
RegiPrep must select one of these four modes:

reformat
  - Apply a 0, 90, 180, or 270 degree clockwise
    rotation about each image axis
  - Flip the order of the voxel grid for any
    of the axes (e.g. dorsal --> ventral
    becomes ventral --> dorsal)
  - Supports multiple file input types
  - Writes Nifti images: nifti.nimh.nih.gov
  - Sets voxel size correctly in metadata
  - Provide an arbitrary number of channel paths
    to --image1

preprocess
  - Prepares two image domains for deformable
    registration simultaneously
  - Requires Nifti images
  - Supports an arbitrary number of channels
    for both image domains
  - Resamples both image domains to a common
    minimum voxel size
  - Estimates a foreground mask for each image
    domain
  - Crops both image domains to a common minimum
    field of view
  - Stores new voxel size and offsets to original
    domain origin in metadata

transfer_metadata
  - Copies metadata from --image1 to --image2
  - Requires Nifti images 
  - Supports one-to-one, one-to-many, and
    many-to-many transfers
  - For many-to-many, --image1 and --image2 must
    have the same number of channels

transfer_preprocessing
  - Applies same resample and crop/pad to
    --image2 as was already applied to --image1
  - Requires Nifti images
  - --image1 must have been preprocessed with
    this tool
  - --image2 channels must have correct metadata
  - Supports one-to-one, one-to-many, and
    many-to-many transfers
  - For many-to-many, --image1 and --image2 must
    have the same number of channels
"""

args_dict = {
  'mode':'which mode to run, see below',
  '--image1':'Filepaths to first image channels',
  '--image2':'Filepaths to second image channels',
  '--outdir':'Where outputs are written; default: execution directory',
  '--pad_size':'background pad added to both sides of all dimensions',
  '--min_vox_size':'minimum voxel size for resampling',
  '--im1_vox_size':'im1 voxel size, required if im1 is not nifti',
  '--im2_vox_size':'im2 voxel size, required if im2 is not nifti',
  '--im1_lambda':'lambda param for foreground segmentation of image1 (smaller -> tighter)',
  '--im2_lambda':'lambda param for foreground segmentation of image2 (smaller -> tighter)',
  '--im1_mask':'Filepath to mask for image1, prevents computation of new mask',
  '--im2_mask':'Filepath to mask for image2, prevents computation of new mask',
  '--im1_rotation':'Rotate im1 clockwise, specify axis (0, 1, or 2) and degs (90, 180, or 270)',
  '--im2_rotation':'Rotate im2 clockwise, specify axis (0, 1, or 2) and degs (90, 180, or 270)',
  '--im1_reorder':'Invert axis order for im 1 (e.g. vent-->dors to dors-->vent), specify axes',
  '--im2_reorder':'Invert axis order for im 2 (e.g. vent-->dors to dors-->vent), specify axes'}

# TODO: consider implementing custom Action subclasses to handle x-delimited lists
#       internally to the parser, and something similar for rotations and reorders
options_dict = {a:{'help':args_dict[a]} for a in args_dict.keys()}
options_dict['mode'] = {**options_dict['mode'],
                        'choices':['reformat',
                                   'preprocess',
                                   'transfer_metadata',
                                   'transfer_preprocessing']}
options_dict['--image1'] =     {**options_dict['--image1'], 'nargs':'+',
                                'required':True}
options_dict['--image2'] =     {**options_dict['--image2'], 'nargs':'*'}
options_dict['--outdir'] =     {**options_dict['--outdir'],
                                'default':abspath(getcwd())}
options_dict['--pad_size'] =   {**options_dict['--pad_size'], 'type':int,
                                'default':5}
options_dict['--im1_lambda'] = {**options_dict['--im1_lambda'], 'type':float,
                                'default':20.}
options_dict['--im2_lambda'] = {**options_dict['--im2_lambda'], 'type':float,
                                'default':20.}

parser = argparse.ArgumentParser(description=desc, epilog=epi,
         formatter_class=RawTextHelpFormatter)
for arg in args_dict.keys():
    parser.add_argument(arg, **options_dict[arg])

# --------------------------- Functions --------------------------------------------
def read_all_channels(filepaths_string, rotate=None, reorder=None, vox_size=None):
    # expected syntax: [[im1_ch1.nii.gz,im1_ch2.nii,...]... ], see mandatory_helplist
    im_paths = [abspath(x.strip('[]')) for x in filepaths_string.split(',')]
    im, im_meta, im_names = [], [], []
    default_nifti_meta = fileio.get_default_nifti_header()
    for path in im_paths:
        if not (path[-4:] == '.nii' or path[-7:] == '.nii.gz') and vox_size is None:
            print('ERROR: If not using nifti format you must specify vox size, run with -h for help')
            sys.exit()
        data, meta = fileio.read_image(path)
        meta = {**default_nifti_meta, **meta}
        if vox_size is not None:
            meta['pixdim'][1:4] = [float(x) for x in vox_size.split('x')]
        data, meta = reformat_data(data, meta, rotate, reorder)
        name = basename(path).split('.')[0]
        im.append(data); im_meta.append(meta); im_names.append(name)
    return im, im_meta, im_names


def reformat_data(im, meta, rotate, reorder):
    # Does not modify q/s forms, as reformat should only be called before
    # any preprocessing or alignment has been done, and thus q/s forms are
    # arbitrary/not set
    # TODO: for multiple axes, there is an ordering effect to multiple rotations
    #       for 2nd and 3rd rotations, I need to compute new axis based on old
    #       axis index and already applied rotations
    if rotate:
        rotations = rotate.split('|')
        for rotation in rotations:
            axis, degrees = [int(x) for x in rotation.split('x')]
            rot_plane = tuple([x for x in [0, 1, 2] if x != axis])
            im = np.rot90(im, degrees/90, rot_plane)
            meta['dim'][1:4] = im.shape
            if degrees/90 == 1 or degrees/90 == 3:
                a, b = meta['pixdim'], rot_plane  # to be succinct
                a[b[0]+1], a[b[1]+1] = a[b[1]+1], a[b[0]+1]
    if reorder:
        axes = tuple([int(x) for x in reorder.split('x')])
        im = np.flip(im, axes)
    return im, meta


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
    im1_list, im1_meta_list, im1_names = read_all_channels(args.image1,
                                                           args.rotate_im1,
                                                           args.reorder_im1,
                                                           args.im1_vox_size)
    im2_list, im2_meta_list, im2_names = read_all_channels(args.image2,
                                                           args.rotate_im2,
                                                           args.reorder_im2,
                                                           args.im2_vox_size)
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




def transfer_preprocessing(args, outdir):

    print("BEGIN PREPROCESSING TRANSFER")
    print("\tREADING IMAGES")
    im1_list, im1_meta_list, im1_names = read_all_channels(args.image1,
                                                           args.rotate_im1,
                                                           args.reorder_im1,
                                                           args.im1_vox_size)
    im2_list, im2_meta_list, im2_names = read_all_channels(args.image2,
                                                           args.rotate_im2,
                                                           args.reorder_im2,
                                                           args.im2_vox_size)
    if not len(im1_list) == 1:
        print("ERROR: Only one reference allowed in transfer mode")
        sys.exit()

    print("\tGETTING PREPROCESSING PARAMETERS")
    """assumed that all images in im2_list have the same shape, origin, and voxel size"""
    d = len(im1_list[0].shape)  # image dimensionality
    origin1 = np.array([im1_meta_list[0]['srow_'+a][-1] for a in ['x', 'y', 'z']])
    origin2 = np.array([im2_meta_list[0]['srow_'+a][-1] for a in ['x', 'y', 'z']])
    v1 = im1_meta_list[0]['pixdim'][1:d+1]
    v2 = im2_meta_list[0]['pixdim'][1:d+1]
    dims1 = np.array(im1_list[0].shape)
    dims2 = np.round(np.array(im2_list[0].shape) * v2/v1).astype(np.int)  # at im1 voxel size
    left_offsets = ( (origin1 - origin2) /v1).astype(int)
    right_offsets = dims1 - dims2 + left_offsets
    box = [slice(slc_positive(l), slc_negative(r)) for l, r in zip(left_offsets, right_offsets)]
    pad = [(-pad_negative(l), pad_positive(r)) for l, r in zip(left_offsets, right_offsets)]

    print("\tAPPLYING PREPROCESSING")
    for im, name in zip(im2_list, im2_names):
        im_pp = pp._resample(im, vox=(v2, v1))
        im_pp = np.pad(im_pp[box], pad, mode='constant')
        fileio.write_image(outdir+'/'+name+'_pp.nii.gz', im_pp, im1_meta_list[0])
    print("\tPREPROCESSING TRANSFER COMPLETE")




def transfer_metadata(args, outdir):

    print("BEGIN METADATA TRANSFER")
    print("\tREADING IMAGES")
    im1_list, im1_meta_list, im1_names = read_all_channels(args.image1,
                                                           args.rotate_im1,
                                                           args.reorder_im1,
                                                           args.im1_vox_size)
    im2_list, im2_meta_list, im2_names = read_all_channels(args.image2,
                                                           args.rotate_im2,
                                                           args.reorder_im2,
                                                           args.im2_vox_size)
    if not len(im1_list) == 1:
        print("ERROR: Only one reference allowed in transfer mode")
        sys.exit()

    print("\tTRANSFERING METADATA INFO")
    for im, name in zip(im2_list, im2_names):
        fileio.write_image(outdir+'/'+name+'_pp.nii.gz', im, im1_meta_list[0])
    print("\tMETADATA TRANSFER COMPLETE")


# TODO: consider consolidating code duplications (e.g. reading/writing files in different modes)
if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    if not isdir(args.outdir):
        try:
            mkdir(args.outdir)
        except OSError as err:
            print("Could not create outdir:\n{0}".format(err), file=sys.stderr)

#    print("DETERMINING MODE")
#    if args.mode == 'reformat':
#        reformat(args)
#    elif args.mode == 'preprocess':
#        preprocess(args)
#    elif args.mode == 'transfer_metadata':
#        transfer_metadata(args)
#    elif args.mode == 'transfer_preprocessing':
#        transfer_preprocessing(args)
#    else:
#        print("TODO: error message pending")
