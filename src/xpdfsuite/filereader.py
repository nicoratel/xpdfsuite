import re
import os
import numpy as np
import h5py


   
def load_h5_data(file, frame='mean', normalize= False, verbose=True):
    """
    Load image data from an HDF5/NXS file using h5py.

    Tries the ID15/ESRF Pilatus structure first:

    * ``<group>/measurement/data``  → image stack (frames × height × width)
    * ``<group>/instrument/pilatus/acquisition/nb_frames``
    * ``<group>/instrument/pilatus/detector_information/pixel_size/{xsize,ysize}``

    Falls back to a generic search: the largest 2-D or 3-D numeric dataset in
    the file is used as the image stack.

    Parameters
    ----------
    file : str
        Path to the ``.h5``, ``.hdf5``, or ``.nxs`` file.
    frame : int or 'mean', optional
        Frame index to return, or ``'mean'`` (default) to average all frames.
    verbose : bool, optional
        Print a summary of detected metadata when ``True``.

    Returns
    -------
    detector_info : dict
        Keys: ``camera_type``, ``camera_title``, ``pixel_size`` (m),
        ``image_width`` (px), ``image_height`` (px), ``binning``,
        ``description``, ``wavelength``, ``nb_frames``.
    raw_image : ndarray
        2-D float array (the selected or averaged frame).

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    ValueError
        If no suitable image dataset is found in the file.
    """
    
    detector_info = {
        'camera_type': None,
        'camera_title': None,
        'pixel_size': None,
        'image_width': None,
        'image_height': None,
        'binning': 1,
        'description': 'HDF5/NXS file',
        'wavelength': None,
        'nb_frames': None,
    }

    with h5py.File(file, 'r') as f:
        group = list(f.keys())[0]
        detector_info['camera_title'] = str(group)
        data = None

        # --- Try ID15 / Pilatus structure ---
        try:
            data = np.array(f[group + '/measurement/data'])

            try:
                ps_path = group + '/instrument/pilatus/detector_information/pixel_size/'
                pixel_size_x = float(f[ps_path + 'xsize'][()])
                pixel_size_z = float(f[ps_path + 'ysize'][()])
                # Values are typically in metres (e.g. 1.72e-4 m for Pilatus)
                detector_info['pixel_size'] = pixel_size_x
            except Exception:
                pass

            try:
                nb_frames = int(f[group + '/instrument/pilatus/acquisition/nb_frames'][()])
                detector_info['nb_frames'] = nb_frames
            except Exception:
                detector_info['nb_frames'] = data.shape[0] if data.ndim == 3 else 1

            if verbose:
                print(f"  ✓ ID15/Pilatus HDF5 structure detected in group '{group}'")

        except Exception:
            pass

        # --- Generic fallback: largest 2-D or 3-D numeric dataset ---
        if data is None:
            best_key = None
            best_size = 0

            def _find_largest(name, obj):
                nonlocal best_key, best_size
                if isinstance(obj, h5py.Dataset):
                    if obj.ndim in (2, 3) and np.issubdtype(obj.dtype, np.number):
                        if obj.size > best_size:
                            best_size = obj.size
                            best_key = name

            f.visititems(_find_largest)

            if best_key is not None:
                data = np.array(f[best_key])
                if verbose:
                    print(f"  ✓ Generic HDF5: using dataset '{best_key}' (shape {data.shape})")
            else:
                raise ValueError(f"No suitable image dataset found in {file}")

    # --- Build 2-D output image ---
    if data.ndim == 3:
        nb = data.shape[0]
        if detector_info['nb_frames'] is None:
            detector_info['nb_frames'] = nb
        if frame == 'mean':
            raw_image = np.mean(data, axis=0).astype(float)
        elif isinstance(frame, int) and 0 <= frame < nb:
            raw_image = data[frame].astype(float)
        else:
            raw_image = np.mean(data, axis=0).astype(float)
    elif data.ndim == 2:
        raw_image = data.astype(float)
        detector_info['nb_frames'] = 1
    else:
        raise ValueError(f"Unexpected dataset shape {data.shape} in {file}")

    detector_info['image_height'], detector_info['image_width'] = raw_image.shape

    if verbose:
        print(f"Loaded H5 file: {file}")
        for key, value in detector_info.items():
            if value is not None:
                print(f"  {key}: {value}")
    if normalize:
        #raw_image -= raw_image.min()
        if raw_image.max() > 0:
            raw_image /= raw_image.max()
    return detector_info, raw_image

