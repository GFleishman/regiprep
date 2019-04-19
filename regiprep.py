#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------- Imports ----------------------------------------------
import sys
from os.path import abspath, isdir, basename, splitext
from os import mkdir
import numpy as np
import fileio
import preprocessing as pp
from interface import parser

# --------------------------- Functions --------------------------------------------
def read_all_channels(im_paths, vox_size=None):
    im, im_meta, im_names = [], [], []
    for path in im_paths:
        filename_pieces = basename(path).split('.')
        if 'nii' not in filename_pieces and vox_size is None:
            print('ERROR: For non-Nifti images you must specify vox size')
            sys.exit()
        data, meta = fileio.read_image(abspath(path))
        def_meta = fileio.get_default_nifti_header(shape=data.shape, dtype=data.dtype)
        meta = {**def_meta, **meta}
        meta['dim'][1:4] = data.shape
        if vox_size is not None:
            meta['pixdim'][1:4] = [float(x) for x in vox_size.split('x')]
        im.append(data); im_meta.append(meta); im_names.append(filename_pieces[0])
    return im, im_meta, im_names


def read_all_swc(im_paths):
    im, im_names = [], []
    for path in im_paths:
        filename_pieces = basename(path).split('.')
        swc = fileio.read_swc(path)
        im.append(swc); im_names.append(filename_pieces[0])
    return im, im_names


# --------------------------- Modes -----------------------------------------------
def reformat(args):
    # Does not modify q/s forms, as reformat should only be called before
    # any preprocessing or alignment has been done, and thus q/s forms are
    # arbitrary/not set
    im1, im1_vs = args.image1, args.im1_vox_size
    im_list, im_meta_list, im_names = read_all_channels(im1, im1_vs)
    for im, meta, name in zip(im_list, im_meta_list, im_names):
        if args.im1_rotation:
            rotations = args.im1_rotation.split('.')
            axes = [int(r.split('x')[0]) for r in rotations]
            degrees = [int(r.split('x')[1]) for r in rotations]
            permute = {0:0, 1:1, 2:2}
            for axis, degree in zip(axes, degrees):
                rot_plane = np.array([x for x in [0, 1, 2] if x != axis])
                permuted_rot_plane = [x for x in [0, 1, 2] if x != permute[axis]]
                im = np.rot90(im, degree/90, permuted_rot_plane)
                if degree/90 % 2 == 1:
                    permute = {axis:permute[axis],
                               rot_plane[0]:permute[rot_plane[1]],
                               rot_plane[1]:permute[rot_plane[0]]}
                    a, b = meta['pixdim'], rot_plane+1 # pixdims essentially 1-indexed
                    a[b[0]], a[b[1]] = a[b[1]], a[b[0]]
                meta['dim'][1:4] = im.shape
        if args.im1_reorder:
            axes = [int(x) for x in args.im1_reorder.split('x')]
            im = np.flip(im, axes)
        fileio.write_image(args.outdir+'/'+name+'_regiprep_reformat.nii.gz', im, meta)


def preprocess(args):

    """Also initializing some helpful variables for later"""
    im1, im1_vs = args.image1, args.im1_vox_size
    im2, im2_vs = args.image2, args.im2_vox_size
    im1_list, im1_meta_list, im1_names = read_all_channels(im1, im1_vs)
    im2_list, im2_meta_list, im2_names = read_all_channels(im2, im2_vs)
    dim = len(im1_list[0].shape)  # image dimension
    v1 = im1_meta_list[0]['pixdim'][1:dim+1]  # im1 voxel size
    v2 = im2_meta_list[0]['pixdim'][1:dim+1]
    im1_brain_mask, im2_brain_mask = None, None
    if args.im1_mask is not None:
        im1_brain_mask, mask_meta = fileio.read_image(args.im1_mask)
    if args.im2_mask is not None:
        im2_brain_mask, mask_meta = fileio.read_image(args.im2_mask)


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
                                    im1_list[0], newvox, min_size=min_size)
    if im2_brain_mask is not None:
        im2_brain_mask, b, nv = pp.normalize_voxelsize(im2_brain_mask, v2,
                                    im2_list[0], newvox, min_size=min_size)


    # TODO: normalize_intensity has 'sigmoid' mode, experiment with that
    # TODO: consider better options here (sum projection in foreground, mean
    #       projection in background - using a quick heuristic foreground/
    #       background threshold)
    #       Use the very nice two-channel image I stitched for Yu/Sujatha
    #       to test alternatives. Do a dual channel alignment to the Z-brain.
    if args.im1_mask is None:
        """This norm is only for preprocessing purposes, don't overwrite im lists"""
        im_list_norm = [pp.normalize_intensity(im) for im in im1_list]
        im_sum = np.sum(np.array(im_list_norm), axis=0)
        im1_brain_mask =  pp.brain_detection(im_sum, v1, foreground_mask=None,
                              iterations=[60, 25, 5], lambda2=args.im1_lambda)
    if args.im2_mask is None:
        im_list_norm = [pp.normalize_intensity(im) for im in im2_list]
        im_sum = np.sum(np.array(im_list_norm), axis=0)
        im2_brain_mask =  pp.brain_detection(im_sum, v2, foreground_mask=None,
                              iterations=[60, 25, 5], lambda2=args.im2_lambda)


    """Padding is after brain detection, so zeros can't affect masking
       Masks will be used for registration, and pads are not within mask
       so pads shouldn't affect registration either"""
    im1_box = pp.minimal_bounding_box(im1_brain_mask)
    im2_box = pp.minimal_bounding_box(im2_brain_mask)
    im1_box_dims = np.array([a.stop - a.start for a in im1_box])
    im2_box_dims = np.array([a.stop - a.start for a in im2_box])
    final_dims = np.maximum(im1_box_dims, im2_box_dims) + 2*args.pad_size
    diff1, diff2 = (final_dims - im1_box_dims)/2., (final_dims - im2_box_dims)/2.
    pad1 = np.array([(np.ceil(d), np.floor(d)) for d in diff1]).astype(int)
    pad2 = np.array([(np.ceil(d), np.floor(d)) for d in diff2]).astype(int)
    for i, (im1, im2) in enumerate(zip(im1_list, im2_list)):
        im1_list[i] = np.pad(im1[im1_box], pad1, mode='constant')
        im2_list[i] = np.pad(im2[im2_box], pad2, mode='constant')
    im1_brain_mask = np.pad(im1_brain_mask[im1_box], pad1, mode='constant')
    im2_brain_mask = np.pad(im2_brain_mask[im2_box], pad2, mode='constant')


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


    sfx = '_regiprep_mask.nii.gz'
    fileio.write_image(args.outdir+'/'+im1_names[0]+sfx, im1_brain_mask, im1_meta_list[0])
    fileio.write_image(args.outdir+'/'+im2_names[0]+sfx, im2_brain_mask, im2_meta_list[0])
    sfx = '_regiprep_pp.nii.gz'
    for i in range(len(im1_list)):
        fileio.write_image(args.outdir+'/'+im1_names[i]+sfx, im1_list[i], im1_meta_list[i])
        fileio.write_image(args.outdir+'/'+im2_names[i]+sfx, im2_list[i], im2_meta_list[i])


def transfer_preprocessing(args):
    # TODO: generalize this function to look for swc files and apply transfer appropriately
    #       need to add a comment line to the beginning of swc files indicating the shift applied to them
    #       # preprocessed with regiprep; shift relative to origin: x-shift, y-shift, z-shift

    im1, im1_vs = args.image1, args.im1_vox_size
    im2, im2_vs = args.image2, args.im2_vox_size

    if '.swc' in im2[0]:
        im1_list, im1_meta_list, im1_names = read_all_channels(im1, im1_vs)
        im2_list, im2_names = read_all_swc(im2)
        if not len(im1_list) == 1:
            print("ERROR: Only one reference allowed in transfer mode")
            sys.exit()

        d = len(im1_list[0].shape)  # image dimensionality
        origin1 = np.array([im1_meta_list[0]['srow_'+a][-1] for a in ['x', 'y', 'z']])
        for im, name in zip(im2_list, im2_names):
            im.translate(im.offset - origin1)
            fileio.write_swc(args.outdir+'/'+name+'_regiprep_tpp.swc', im)

    else:
        im1_list, im1_meta_list, im1_names = read_all_channels(im1, im1_vs)
        im2_list, im2_meta_list, im2_names = read_all_channels(im2, im2_vs)
        if not len(im1_list) == 1:
            print("ERROR: Only one reference allowed in transfer mode")
            sys.exit()

        """assumed that all images in im2_list have the same shape, origin, and voxel size"""
        d = len(im1_list[0].shape)  # image dimensionality
        origin1 = np.array([im1_meta_list[0]['srow_'+a][-1] for a in ['x', 'y', 'z']])
        origin2 = np.array([im2_meta_list[0]['srow_'+a][-1] for a in ['x', 'y', 'z']])
        v1 = im1_meta_list[0]['pixdim'][1:d+1]
        v2 = im2_meta_list[0]['pixdim'][1:d+1]
        dims1 = np.array(im1_list[0].shape)
        dims2 = np.round(np.array(im2_list[0].shape) * v2/v1).astype(np.int)  # at im1 voxel size
        left_offsets = ( (origin1 - origin2) /v1 ).astype(int)
        right_offsets = dims1 - dims2 + left_offsets
        f1 = lambda x: x if x > 0 else None
        f2 = lambda x: x if x < 0 else None
        box = [slice(f1(l), f2(r)) for l, r in zip(left_offsets, right_offsets)]
        f1 = lambda x: abs(x) if x < 0 else 0
        f2 = lambda x: x if x > 0 else 0
        pad = [(f1(l), f2(r)) for l, r in zip(left_offsets, right_offsets)]

        for im, name in zip(im2_list, im2_names):
            im_pp = pp._resample(im, vox=(v2, v1))
            im_pp = np.pad(im_pp[box], pad, mode='constant')
            fileio.write_image(args.outdir+'/'+name+'_regiprep_tpp.nii.gz', im_pp, im1_meta_list[0])


def transfer_metadata(args):

    im1, im1_vs = args.image1, args.im1_vox_size
    im2, im2_vs = args.image2, args.im2_vox_size
    im1_list, im1_meta_list, im1_names = read_all_channels(im1, im1_vs)
    im2_list, im2_meta_list, im2_names = read_all_channels(im2, im2_vs)
    if not len(im1_list) == 1:
        print("ERROR: Only one reference allowed in transfer mode")
        sys.exit()

    for im, name in zip(im2_list, im2_names):
        fileio.write_image(args.outdir+'/'+name+'_regiprep_tm.nii.gz', im, im1_meta_list[0])


# --------------------------- Entry Point -----------------------------------------
if __name__ == '__main__':

    args = parser.parse_args()
    if args.verbose:
        print("RegiPrep arguments:")
        for key in args.__dict__.keys():
            print(key, ':\t', args.__dict__[key])

    if not isdir(args.outdir):
        try:
            mkdir(args.outdir)
        except OSError as err:
            print("Could not create outdir:\n{0}".format(err), file=sys.stderr)

    modes = {'reformat':reformat,
             'preprocess':preprocess,
             'transfer_metadata':transfer_metadata,
             'transfer_preprocessing':transfer_preprocessing}
    modes[args.mode](args)

