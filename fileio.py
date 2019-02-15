#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------- Imports ----------------------------------------------
# TODO: h5py and nibabel have annoying future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from os.path import basename, splitext, abspath
import glob
import sys


# --------------------------- Constants --------------------------------------------
# TODO: define mappings between essential meta data field names, ensure
# they are read and stored in meta_dict
_dtypes_dict = { 8: np.uint8,
                16: np.uint16,
                32: np.float32,
                64: np.float64}
_compression_extensions = ['.gz']
EXTL_LINK_KEY = '_EXTLINK'
REGREF_ATTR_KEY = '_REGREF'
SLICE_ATTR_KEY = 'REGION_ARRAY'
CHUNK_SIZE = 128
CHUNK_NUM = 10


# --------------------------- Interal Functions ------------------------------------
# UTILITIES
def _get_extension(path):
    extension = splitext(basename(path))[1]
    if extension in _compression_extensions:
        extension = splitext(splitext(basename(path))[0])[1] + extension
    return extension

def _string_to_slice(slice_string):
    # Helper function for slice string parsing
    indices = slice_string.split(':')
    if len(indices) == 1:
        return slice(int(indices[0]), int(indices[0])+1, None)
    else:
        return slice(*map(lambda x: int(x) if x else None, indices))

def _standardize_axis_order(array, img_type):
    # TODO: are these magic nums, or do libs always load with these orderings?
    # is there a better way to ensure writing occurs in canonical orientation?
    positions = np.argsort(array.shape)[::-1]
    if img_type == 'tiff':
        new_positions = [0, 1, 2]
    elif img_type == 'hdf5':
        new_positions = [2, 1, 0]
    elif img_type == 'nii':
        new_positions = [0, 1, 2]
    elif img_type == 'n5':
        new_positions = [2, 1, 0]
    return np.moveaxis(array, positions, new_positions)


# TIFF
def _read_tiff(path, meta_dict={}):
    from PIL import Image
    from PIL.TiffTags import TAGS
    image = Image.open(path)
    img_meta_dict = {TAGS[key] : image.tag[key] for key in image.tag.keys()}
    meta_dict = {**img_meta_dict, **meta_dict}
    h, w, d = image.height, image.width, image.n_frames
    dtype = _dtypes_dict[meta_dict['BitsPerSample'][0]]
    image_data = np.empty((h, w, d), dtype=dtype)
    for i in range(d):
        image.seek(i)
        image_data[..., i] = np.array(image)
    image.close()
    return image_data, meta_dict

def _write_tiff(path, image_data, meta_dict={}):
    from PIL import Image
    from PIL.TiffTags import TAGS, lookup
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    image_data = _standardize_axis_order(image_data, 'tiff')
    h, w, d = image_data.shape
    imgList = []
    # TODO: data type should be determined by input image_data
    # this complicated bit ensures data is saved as 16bit
    for i in range(d):
        imgList.append(Image.new('I;16', image_data.shape[:-1]))
        slc = image_data[..., i]
        slice_bytes = slc.astype(slc.dtype.newbyteorder('<')).tobytes(order='F')
        imgList[i].frombytes(slice_bytes)
    # hack to invert TAGS dictionary, does not support non one-to-one dicts
    TAGS_keys, TAGS_values = list(TAGS.keys()), list(TAGS.values())
    TAGSinv = {v : k for v, k in zip(TAGS_values, TAGS_keys)}
    img_meta_dict = ImageFileDirectory_v2()
    for key in meta_dict.keys():
        if key in TAGSinv.keys():
            tag_info = lookup(TAGSinv[key])
            img_meta_dict[tag_info.value] = meta_dict[key]
            img_meta_dict.tagtype[tag_info.value] = tag_info.type
    imgList[0].save(path, save_all=True, append_images=imgList[1:],
                    compression='tiff_deflate', tiffinfo=img_meta_dict)


# HDF5
def _read_hdf5(path, dataset_path, meta_dict={}):
    import h5py
    with h5py.File(path, 'r') as f:
        image_data = f[dataset_path][...]
    for key in meta_dict.keys():
        if not meta_dict[key] and f.attrs[key]:
            meta_dict[key] = f.attrs[key]
        elif not meta_dict[key] and f[dataset_path].attrs[key]:
            meta_dict[key] = f[dataset_path].attrs[key]
    return image_data, meta_dict

def _write_hdf5(path, dataset_path, image_data, meta_dict={}):
    import h5py
    image_data = _standardize_axis_order(image_data, 'hdf5')
    with h5py.File(path, 'w') as f:
        f[dataset_path] = image_data
        for key in meta_dict.keys():
            if meta_dict[key] is not None:
                f.attrs[key] = meta_dict[key]


# NIFTI
def _read_nifti(path, meta_dict={}):
    import nibabel as nib
    image = nib.load(path)
    image_data = image.get_data().squeeze()
    new_meta_dict = dict(image.header)
    meta_dict = {**new_meta_dict, **meta_dict}
    return image_data, meta_dict

def _write_nifti(path, image_data, meta_dict={}):
#    image_data = _standardize_axis_order(image_data, 'nii') # possibly a bad idea
    import nibabel as nib
    image = nib.Nifti1Image(image_data, None)
    for key in meta_dict.keys():
        if key in image.header.keys():
            image.header[key] = meta_dict[key]
    nib.save(image, path)


# N5
def _read_n5(path, dataset_path, meta_dict={}):
    import z5py
    with z5py.File(path) as f:
        image_data = f[dataset_path][...]
    for key in meta_dict.keys():
        if not meta_dict[key] and f.attrs[key]:
            meta_dict[key] = f.attrs[key]
        elif not meta_dict[key] and f[dataset_path].attrs[key]:
            meta_dict[key] = f[dataset_path].attrs[key]
    return image_data, meta_dict

def _write_n5(path, dataset_path, image_data, meta_dict={},
              chunk_method=1, chunks=()):
    import z5py
    image_data = _standardize_axis_order(image_data, 'n5')
    if chunk_method == 1:
        chunks = [min([s, CHUNK_SIZE]) for s in image_data.shape]
    elif chunk_method == 2:
        chunks = [s//CHUNK_NUM for s in image_data.shape]
    elif chunk_method == 3:
        if not chunks:
            print("If chunk_method=3 then chunks must be a valid chunk shape")
            sys.exit()
    image = z5py.File(path)
    ds = image.create_dataset(dataset_path, image_data.dtype,
                              shape=image_data.shape, chunks=chunks,
                              compression='gzip', level=1, n_threads=8)
    ds[...] = image_data
    for key in meta_dict.keys():
            if meta_dict[key] is not None:
                # TODO: following test is temporary hack, need to deal with
                # meta data compatibility
                import numbers
                if (isinstance(meta_dict[key], str) or 
                    isinstance(meta_dict[key], numbers.Number)): 
                    image.attrs[key] = meta_dict[key]


# HDF5 EXTERNAL LINK CONTAINER
def _write_hdf5_ext_link_container(files_to_wrap, dataset_path, path,
                                   space_slices, meta_dict={}):
    import h5py
    space_slices_arr = np.array([[s.start, s.stop, s.step] for s in space_slices])
    # get image dimensions from first example, this assumes all images
    # have the same dimensions
    with h5py.File(files_to_wrap[0], 'r') as f:
        for i, dim in enumerate(f[dataset_path].shape):
            if not space_slices_arr[i, 1]: space_slices_arr[i, 1] = dim
    space_slices_arr[space_slices_arr == None] = 0
    space_slices_arr = space_slices_arr.astype(np.int16)
    container = h5py.File(path, 'w')
    for i, h5_file in enumerate(files_to_wrap):
        key = splitext(basename(h5_file))[0] + EXTL_LINK_KEY
        container[key] = h5py.ExternalLink(h5_file, dataset_path)
        if 'regref' in meta_dict.keys():
            with h5py.File(h5_file, 'r+') as f:
                regref = f[dataset_path].regionref[tuple(space_slices)]
                container.attrs[key+REGREF_ATTR_KEY] = regref
        for k in meta_dict.keys():
            if meta_dict[k] is not None:
                container.attrs[k] = meta_dict[k]
        container.attrs[SLICE_ATTR_KEY] = space_slices_arr


# --------------------------- External Functions -----------------------------------
# TODO: add docstrings
def read_image(path, meta_dict={}, h5_dataset_path=''):
    file_format = _get_extension(path)
    path = abspath(path)
    if file_format in ['.tiff', '.tif']:
        return _read_tiff(path, meta_dict)
    elif file_format in ['.h5']:
        return _read_hdf5(path, h5_dataset_path, meta_dict)
    elif file_format in ['.nii', '.nii.gz']:
        return _read_nifti(path, meta_dict)
    elif file_format in ['.n5']:
        return _read_n5(path, h5_dataset_path, meta_dict)


def write_image(path, image_data, meta_dict={}, h5_dataset_path=''):
    file_format = _get_extension(path)
    path = abspath(path)
    if file_format in ['.tiff', '.tif']:
        _write_tiff(path, image_data, meta_dict)
    elif file_format in ['.h5']:
        _write_hdf5(path, h5_dataset_path, image_data, meta_dict)
    elif file_format in ['.nii', '.nii.gz']:
        _write_nifti(path, image_data, meta_dict)
    elif file_format in ['.n5']:
        _write_n5(path, h5_dataset_path, image_data, meta_dict)


def convert_image(in_path, out_path, meta_dict={}, h5_dataset_path=''):
    image_data, meta_dict = read_image(in_path, meta_dict, h5_dataset_path)
    write_image(out_path, image_data, meta_dict, h5_dataset_path)


def make_hdf5_ext_link_container(in_path, out_path, meta_dict={},
                                 h5_dataset_path='', slice_=None):
    # TODO: test if in_path is a directory, if it is, test if it ends with a '/'
    # if it does do nothing, if it doesn't add '/'
    all_h5_files = glob.glob(in_path+"*.h5")
    all_h5_files = sorted(list(map(abspath, all_h5_files)))
    if slice_:
        slices = [_string_to_slice(x.strip('[]')) for x in slice_.split(',')]
        space_slices, time_slice = slices[:-1], slices[-1]
    _write_hdf5_ext_link_container(all_h5_files[time_slice], h5_dataset_path,
                                   out_path, space_slices, meta_dict)


def get_default_nifti_header():
    import nibabel as nib
    img = nib.Nifti1Image(np.reshape(np.arange(8), (2,2,2)), np.eye(4))
    return img.header

