import nibabel as nb
import numpy as np
import pandas as pd
from nibabel import processing as nbproc
import nilearn
from scipy import spatial
from nipype.interfaces import ants
import deepdish
import re
import os

def epi_lesion_confound(lesion_fn, melodic_fn, brain_mask_fn, thresh = 0.05):
    '''
    Finds melodic components with a high degree of overlap with lesion mask,
    writes regressors for cleaning. Method outlined in
    Yourganov et al. (2018). Removal of artifacts from resting-state fMRI data in stroke, Neuroimage Clin

    Input:
    lesion_fn - Path to lesion mask
    melodic_fn - Path to melodic 4d output (found in working directory of fmriprep)
    Note: This image is in 2mm space. Gets resliced to lesion mask dims as part of the process. You're welcome.
    brain_mask_fn - Path to brain mask in functional space.
    thresh - Threshold for similarity between lesion and component (default 0.05)

    Output:
    out_name - Path to csv file with components showing a high degree of overlap with lesion mask.
    '''

    out_path, out_name = os.path.split(lesion_fn)
    out_name = '{}{}'.format(out_name.split('space')[0], 'LesionICs.csv')

    lesion_hdr = nb.load(lesion_fn)
    lesion_im = lesion_hdr.get_fdata()

    ica_hdr = nb.load(melodic_fn)
    ica_resampled = nilearn.image.resample_to_img(ica_hdr, lesion_hdr) #Reslice ica to lesion dimensions
    ica_resampled_im = ica_resampled.get_fdata()

    mask = nb.load(brain_mask_fn).get_fdata().astype(bool)

    c_ji = [] #Component Jaccard Index

    for n in range(0, ica_resampled_im.shape[-1]):
        comp = ica_resampled_im[:, :, :, n].copy()
        comp_flat = comp[mask]

        dist_thresh = np.percentile(abs(comp_flat), 97.5)
        comp_bool = (abs(comp_flat) >= dist_thresh).astype(int)
        lesion_bool = (lesion_im[mask] > 0.1).astype(int) #Should be binary mask. This reduces any possible effects of normalisation (even though it's using NN...)

        c_ji.append(1 - spatial.distance.jaccard(comp_bool, lesion_bool))

    c_ji = np.array(c_ji)

    remove_comps = pd.DataFrame(np.where(c_ji >= thresh)[0]).T #Quick and easy method to save in same format as fmriprep
    print('Identified {} components associated with lesion'.format(remove_comps.size))

    out_path, file_name = os.path.split(lesion_fn)
    out_fn = '{}{}'.format(file_name.split('space')[0], 'LesionICs.csv')
    out_name = os.path.join(out_path, out_fn)
    remove_comps.to_csv(out_name, sep = ',', header = False, index = False)

    return(out_name)


def epi_frist24(fmriprep_confounds_fn):
    '''
    Calculate the Taylor expansion of motion parameters ala Friston et al (year)

    Inputs:
    fmriprep_confouds_fn - Path to fmriprep confounds tsv file.

    Outputs:
    motion24 -  numpy array of 24 motion paramaters
    '''

    out_path, in_fn = os.path.split(fmriprep_confounds_fn)
    out_fn = in_fn.split('desc-')[0] + 'desc-motion24.csv'


    #CALCULATE FRISTON 24 MODEL (6 motion params + preceeding vol + each values squared.)
    motion_df = pd.read_csv(fmriprep_confounds_fn, sep = '\t')
    motion_params = motion_df[['trans_x', 'trans_y', 'trans_z','rot_x', 'rot_y', 'rot_z']].values
    motion_squared = motion_params ** 2
    new_motion = np.concatenate((motion_params, motion_squared), axis = 1)
    motion_roll = np.roll(motion_params, 1, axis = 0)
    motion_roll[0] = 0
    new_motion = np.concatenate((new_motion, motion_roll), axis = 1)
    motion_roll_squared = motion_roll ** 2
    motion24 = np.concatenate((new_motion, motion_roll_squared), axis = 1)

    out_name = os.path.join(out_path, out_fn)
    np.savetxt(out_name, motion24, delimiter = ',')

    return(out_name)



def epi_collect_ICA_confounds(mixing_matrix_fn, noise_comps_fn, lesion_comps_fn  = None):
    '''
    Collects motion and lesion confounds using information in fmriprep output file <subj_id>_task-rest_AROMAnoiseICs.csv
    and <subj_id>_task-rest_LesionICs.csv (if present).

    Extracts these from the mixing matrix file <subj_id>_task-rest_desc-MELODIC_mixing.tsv and writes to a new file.

    Input:
    mixing_matrix_fn - Path to mixing matrix
    noise_comps_fn - Path to AROMA identified noise components
    lesion_comps_fn - Path to epi_lesion_confound identified lesion components

    Output:
    out_name - Path to csv file with identified IC noise components
    '''

    out_path, in_fn = os.path.split(mixing_matrix_fn)
    out_fn = in_fn.split('desc-')[0] + 'desc-ICA_confounds.csv'

    mm = np.genfromtxt(mixing_matrix_fn, delimiter = '\t')
    n_comps = mm.shape[1]

    comp_idx = np.genfromtxt(noise_comps_fn, delimiter = ',').astype(int) -1 #ICA components start at 1, python is 0 indexed.
    if lesion_comps_fn: #Check for overlap between AROMA and lesion identified components
        lesion_idx = np.genfromtxt(lesion_comps_fn, delimiter = ',').astype(int)
        merge_idx = np.append(motion_idx, lesion_idx)
        comp_idx = np.unique(merge_idx)

    n_noise_comps = comp_idx.shape[0]
    aroma_confounds = mm[:, comp_idx]

    print('Number of motion components to remove is: {} of {} ({}%)\n{}'.format(n_noise_comps,
                                                                            n_comps,
                                                                            (n_noise_comps / n_comps) * 100,
                                                                            comp_idx))
    out_name = os.path.join(out_path, out_fn)
    np.savetxt(out_name, aroma_confounds, delimiter = ',')

    return(out_name)



def epi_gen_confounds_matrix(fmriprep_confounds_fn, confound_list):
    '''
    Create the confounds matrix for regression using non-melodic output from fmriprep.

    Inputs:
    fmriprep_confouds_fn - Path to fmriprep confounds tsv file.
    confound_list - List of confounds to be used (note: must match column names in fmriprep confounds tsv)
    '''

    out_path, in_fn = os.path.split(fmriprep_confounds_fn)
    out_fn = in_fn.split('desc-')[0] + 'desc-fmriprep_seleced_confounds.csv'

    df = pd.read_csv(fmriprep_confounds_fn, sep = '\t')
    confounds_vals = df[confound_list].values

    out_name = os.path.join(out_path, out_fn)
    np.savetxt(out_name, confounds_vals, delimiter = ',')
    return(out_name)




def epi_clean(epi_fn, mask_fn, selected_confounds_fn, aroma_confounds_fn, add_const = True, output_z = False):
    '''
    Uses OLS GLM to remove confounds from functional time series. Follows the fsl_regfilt method of
    demeaning functional and confounds, then adding mean back to functional after regression.

    Inputs:
    epi_fn - Path to epi file
    mask_fn - Path to binary brain mask
    selected_confounds_fn - Path to output file from epi_gen_confounds_matrix
    confounds_fn - Path to text file (no delimiters) with timeseries of each confound as a column
    output_z - Bool. Output z scored, cleaned epi image also

    Outputs:
    cleaned_im - Epi with confounds removed
    cleaned_im_z - As above, but z scored.
    '''

    fmriprep_confounds = np.genfromtxt(selected_confounds_fn, delimiter = ',')
    aroma_confounds = np.genfromtxt(aroma_confounds_fn, delimiter = ',')

    if add_const:
        lin = np.arange(0, fmriprep_confounds.shape[0])
        lin = lin[:, np.newaxis]
        X = np.hstack((fmriprep_confounds, aroma_confounds, lin))

    else:
        X = np.hstack((fmriprep_confounds, aroma_confounds))

    X_demeaned = X - X.mean()


    print('Removing {} confounds'.format(X_demeaned.shape[-1]))

    hdr = nb.load(epi_fn)
    im = hdr.get_fdata()

    brain_mask = nb.load(mask_fn).get_fdata().astype(bool)

    y = im[brain_mask].T
    y_mean = im[brain_mask].T.mean()
    y_demeaned = y - y_mean

    b = np.linalg.lstsq(X_demeaned, y_demeaned, rcond =  None)[0]
    y_resid = y - X.dot(b)
    y_resid = y_resid + y_mean

    #Replaces min / max intensities from header with 0 (similar to fsl GLM)
    hdr.header['cal_min'] = 0
    hdr.header['cal_max'] = 0

    cleaned_im = np.zeros_like(im)
    cleaned_im[brain_mask] = y_resid.T

    out_path, in_fn = os.path.split(epi_fn)
    out_fn = in_fn.split('.')[0].split('bold')[0] + 'cleaned.nii.gz'
    out_name = os.path.join(out_path, out_fn)

    nb.Nifti1Image(cleaned_im, header = hdr.header, affine = hdr.affine).to_filename(out_name)

    if output_z:
        del cleaned_im #Save RAM

        cleaned_im_z = np.zeros_like(im)
        cleaned_im_z[brain_mask] = (y_resid.T - y_resid.T.mean()) / y_resid.T.std()

        out_fn_z = in_fn.split('.')[0].split('bold')[0] + 'cleaned_z.nii.gz'
        out_name_z = os.path.join(out_path, out_fn_z)

        nb.Nifti1Image(cleaned_im_z, header = hdr.header, affine = hdr.affine).to_filename(out_name_z)

    return(out_name, out_name_z, X_demeaned if output_z else out_name, X_demeaned)




def epi_smooth(epi_fn, mask_fn, fwhm = None):
    '''
    Smooth epi with a Gaussian kernel

    Inputs:
    epi_fn - Path to epi file
    fwhm - int. Full width half maximum (in mm) of kernel to smooth epi file

    Outputs:
    smoothed_epi - Smoothed epi file to fwhm
    '''

    hdr = nb.load(epi_fn)

    brain_mask = nb.load(mask_fn).get_fdata().astype(bool)

    if not fwhm:
        fwhm = 0
        print('\n\n***WARNING***\nKernel FWHM not set!\n\n')

    print('Smoothing with kernel size: {}'.format(fwhm))

    smoothed_flat = nbproc.smooth_image(img = hdr, fwhm = fwhm).get_fdata()[brain_mask]
    smoothed_epi = np.zeros_like(hdr.get_fdata())
    smoothed_epi[brain_mask] = smoothed_flat

    out_path, in_fn = os.path.split(epi_fn)
    out_fn = in_fn.split('.')[0].split('bold')[0] + '_smoothed_{}mm.nii.gz'.format(fwhm)
    out_name = os.path.join(out_path, out_fn)

    nb.Nifti1Image(smoothed_epi, header = hdr.header, affine = hdr.affine).to_filename(out_name)

    return(out_name)



def anat_apply_transforms(moving_fn, ref_fn, transform_fn, trans_type = 'desc'):
    '''
    Applies transforms calculated by ANTS

    Input:
    moving_fn - Path to file to be transformed
    ref_fn - Path to reference image (eg. if transform is from subj > MNI, reference is MNI template)
    transform_fn - The transform output from ANTs (usually in h5 format)
    trans_type - String of either desc or label. Desc denotes an anatomical scan, label a labelled image (eg. lesion mask, tissue prob mask)

    Output:
    out_fn - Path to transformed image
    '''

    space = re.findall('to-((.*)_(.*_))', transform_fn)[0][1] #Find which space transform is to
    out_path, trans_name = os.path.split(transform_fn) #Places output in same directory as transform file

    in_fn = os.path.split(moving_fn)[-1]
    if trans_type == 'desc':
        pre_split, post_split = in_fn.split('desc-')
        out_fn = pre_split + 'space-' + space + '_desc-' + post_split

    elif trans_type == 'label':
        pre_split, post_split = in_fn.split('label-')
        out_fn = pre_split + 'space-' + space + '_label-' + post_split

    else:
        raise Exception('Only desc or label identifiers are currently supported')

    out_name = os.path.join(out_path, out_fn)
    print(out_name)

    subj2mni = ants.ApplyTransforms()
    subj2mni.inputs.input_image = moving_fn
    subj2mni.inputs.reference_image = ref_fn
    subj2mni.inputs.transforms = transform_fn
    if trans_type == 'desc':
        subj2mni.inputs.interpolation = 'Linear'
    else:
        subj2mni.inputs.interpolation = 'NearestNeighbor' #Don't want mask smoothed!
    subj2mni.inputs.output_image = out_name

    return(out_name)


def anat_gm_mask(dseg_fn, vox_size = None):
    '''
    Uses dseg (tissue segmentation) output from freesurfer and creates a grey matter mask.

    Input:
    dseg_fn - Path to anatomical space dseg
    vox_size - Optional resample in mm (can be int or list of ints)

    Output:
    out_name - Path to grey matter mask
    out_name_ds - Path to downsampled grey matter mask (typically from anatomical to functional)
    '''

    out_path, in_fn = os.path.split(dseg_fn)
    out_fn = in_fn.split('dseg')[0] + 'desc-gm_mask.nii.gz'

    hdr = nb.load(dseg_fn)
    dseg_im = hdr.get_data()
    gm_mask = dseg_im == 2

    out_name = os.path.join(out_path, out_fn)
    nb_image = nb.Nifti1Image(gm_mask, header = hdr.header, affine = hdr.affine)
    nb_image.to_filename(out_name)

    if vox_size:
        ds = nb.processing.resample_to_output(hdr, vox_size)
        ds_data = ds.get_data() == 2
        if type(vox_size) == list:
            vox = vox_size[0]
        else:
            #Figure what's going on below...
            vox = vox_size
        out_fn_ds = in_fn.split('dseg')[0] + 'desc-gm_mask_voxsize_{}mm.nii.gz'.format(vox_size)
        out_name_ds = os.path.join(out_path, out_fn_ds).replace('anat', 'func')
        nb.Nifti1Image(ds_data, header = ds.header, affine = ds.affine).to_filename(out_name_ds)


    return(out_name, out_name_ds if vox_size else out_name)


def anat2epi_ds(anat_fn, epi_3d_fn):
    '''
    Downsamples anatomical image to the size of the functional image
    Note: This is used primarily for downsampling lesion masks to functional space

    Input:
    anat_fn - Path to anatomical image to downsample to functional space
    epi_3d_fn - Path to 3d epi image that contains shape information. Could have been 4d, but I was lazy.

    Output:
    out_name - Path to downsampled anatomical image in functional dimensions
    '''

    out_path, epi_name = os.path.split(epi_3d_fn) #Places output in func directory
    task = re.findall('task-(.*?_)', epi_3d_fn)[0][:-1] #output is TaskType_
    print(task)

    epi_hdr = nb.load(epi_3d_fn)
    vox_size = np.average(epi_hdr.header.get_zooms()).astype(int)

    in_fn = os.path.split(anat_fn)[-1]

    item_list = in_fn.split('T1w')
    item_list.insert(1, 'task-{}'.format(task))
    out_fn = ''.join(item_list)

    hdr = nb.load(anat_fn)

    out_name = os.path.join(out_path, out_fn)

    #Downsample
    anat_im_ds = nb.processing.resample_to_output(hdr, vox_size)
    nb.Nifti1Image(anat_im_ds.get_fdata(), header = anat_im_ds.header, affine = anat_im_ds.affine).to_filename(out_name)

    return(out_name)


def epi_parcellation(epi_fn, parc_fn, parc_ids_fn, atlas_name = 'aal'):

    '''
    Produces parcellation of functional data.
    Note: Functional and parcellation files must be the same dimensions.

    Input:
    epi_fn - Path to functional file to be parcellated
    parc_fn - Path to parcellation file. Should be a single 3d volume with unique IDs per parcellation
    parc_ids_fn - Path to text file with either numerical or anatomical labels for each parcellation
    atlas_name - Name of the atlas used for parcellataion

    Output:
    parc_vox_fn - Path to dictionary (in hdf5 form) that contains each voxel's time series from a parcellation
    parc_ts_fn - Path to .tsv file with mean timeseries from each parcellation region.
    '''

    out_fn_prefix = epi_fn.split('space')[0]
    out_fn_suffix = 'atlas-{}_timeseries'.format(atlas_name)
    out_fn_stem = '{}{}'.format(out_fn_prefix, out_fn_suffix)

    parc_vox = {}
            
    with open(parc_ids_fn) as infile:
    parc_ids = [line.strip().lower() for line in infile]

    hdr = nb.load(epi_fn)
    epi_data = hdr.get_fdata()

    parc = nb.load(parc_fn).get_fdata()
    parc_vals = np.unique(parc)[1:].astype(int)

    id_zip = zip(parc_vals, parc_ids)

    if epi_data.shape[:-1] != parc.shape:
        raise Exception('EPI and parcellation must be in same space')
    else:
        parc_ts = {}

    for n, (parc_n, parc_id) in enumerate(id_zip):
        roi_ts = epi_data[(parc == parc_n)]

        parc_vox[parc_id] = roi_ts
        parc_ts[parc_id] = roi_ts.mean(0)

    parc_vox_fn = '{}_parc_vox.h5'.format(out_fn_stem)
    parc_ts_fn = '{}_mean.tsv'.format(out_fn_stem)

    deepdish.io.save(parc_vox_fn, parc_vox)
    pd.DataFrame(parc_ts).to_csv(parc_ts_fn, sep = '\t', index = False)

    return(parc_ts_fn)


def parc_lesion_overlap(parc_fn, lesion_fn, parc_ids_fn, atlas_name = 'aal'):
    '''
    Calculates overlap between lesion mask and parcellation atlas

    Input:
    parc_fn - Path to parcellation file. Should be a single 3d volume with unique IDs per parcellation
    lesion_fn - Path to lesion mask
    parc_ids_fn - Path to text file with either numerical or anatomical labels for each parcellation
    atlas_name - Name of the atlas used for parcellataion

    Output:
    out_fn - Path to tsv file showing amount of damage per parcel in sum of voxels and percentage of voxels
    '''

    out_fn_prefix = lesion_fn.split('space')[0]
    out_fn_suffix = 'atlas-{}_desc-lesion_damage.tsv'.format(atlas_name)
    out_fn = '{}{}'.format(out_fn_prefix, out_fn_suffix)

    parc_ids = []
    with open(parc_ids_fn) as ids:
        for line in ids:
            parc_ids.append(line[:-1].lower()) #Remove newline

    parc = nb.load(parc_fn).get_fdata()
    parc_vals = np.unique(parc)[1:].astype(int)

    lesion = nb.load(lesion_fn).get_fdata()

    if lesion.shape != parc.shape:
        raise Exception('Lesion mask and parcellation must be in same space')
    else:
        parc_damage = {}

    id_zip = zip(parc_vals, parc_ids)

    for n, (parc_n, parc_id) in enumerate(id_zip):
        parc_size = (parc == parc_n).sum()
        parc_damage[parc_id] = [int(lesion[parc == parc_n].sum()), abs(round(lesion[parc == parc_n].sum() / parc_size, 2))]

    parc_damage_df = pd.DataFrame(parc_damage).T
    parc_damage_df.columns = ['parc', 'sum', 'percent']
    parc_damage_df.to_csv(out_fn, sep = '\t')

    return(out_fn)


def epi_calc_lag(epi_fn, gm_mask_fn, brain_mask_fn):
    '''
    Compute BOLD lag map (ala *insert paper here*).
    '''

    brain_mask = nb.load(brain_mask_fn).get_data().astype(bool)
    gm_mask = nb.load(out_name_ds).get_data().astype(bool)

    hdr = nb.load(epi_fn)
    epi_im = hdr.get_data()
    epi_ts = epi_im[brain_mask]
    tr = hdr.header.get_zooms()[-1]

    global_ts = epi_im[gm_mask].mean(0)

    epi_cross_corr = np.zeros(epi_ts.shape[0])
    nsamples = 1000


    out_path, in_fn = os.path.split(epi_fn)
    out_fn = in_fn.split('desc')[0] + 'desc-lag_map.nii.gz'


    for n, vox in enumerate(epi_ts):
        vox_zero_mean = vox - vox.mean(axis = 0)
        global_ts_zero_mean = global_ts - global_ts.mean(axis = 0)

        #Full cross correlation
    #     c = np.correlate(vox_zero_mean, global_ts_zero_mean, 'same') #Correlation of all shifts of time series.
        c = np.correlate(vox, global_ts, 'full') #Correlation of all shifts of time series.
        zero_lag = np.floor(c.shape[0] / 2).astype(int)
        c_culled = c[zero_lag - 3: zero_lag + 4] #Capture +- 3 TRs (9 seconds)
        maxC_idx = np.where(c == c_culled.max())[0].item() #Find where the maximum value of the windowed correlation exists within the larger cross correlation

        #Sectioned
        y = c[maxC_idx - 1: maxC_idx + 2] #Collect max +- 1 points for polynomial fit
        x = np.arange(0, len(y)) #Split 2d array into 1d
        z = np.polyfit(x, y, 2) #Second decree (parabolic) polynomial fit w/ coefs
        p = np.poly1d(z) # Calculate discrete vals
        xp = np.linspace(0, 2, nsamples) # Create array to house sample values
        interp = p(xp) # Calculate nsamples of poly func
        x_new = np.linspace(tr * -1, tr, p(xp).shape[0]) # Generate lag time series
        maxP_idx = np.where(interp == interp.max())[0].item() #Get max lag from interpolated time series

        #Putting it all back together
        lag_point = zero_lag - maxC_idx
        lag_point_seconds = tr * lag_point
        lag_val = np.round(lag_point_seconds + x_new[maxP_idx], 2)

        epi_cross_corr[n] = lag_val

    lag_map = np.zeros_like(brain_mask.astype(float))
    lag_map[brain_mask] = epi_cross_corr

    out_name = os.path.join(out_path, out_fn)

    lag_nii = nb.Nifti1Image(lag_map, header = hdr.header, affine = hdr.affine)
    lag_nii.to_filename(out_name)

    return(out_name)
