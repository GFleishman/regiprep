from os import getcwd
from os.path import abspath
import argparse
from argparse import RawTextHelpFormatter


# --------------------------- Interface Specs --------------------------------------
desc = """
~~***~~**~*   RegiPrep   *~**~~***~~~
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
