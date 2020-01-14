import os
import glob
from confounds_funcs import epi_lesion_confound, epi_frist24, epi_collect_ICA_confounds, epi_gen_confounds_matrix, epi_clean, epi_smooth, anat_apply_transforms, anat_gm_mask, anat2epi_ds, epi_parcellation, parc_lesion_overlap, epi_calc_lag
import sys

#Set up paths / subj level info / input files

#Path to where data is kept
subj_id = sys.argv[1]
root_dir = sys.argv[2]
raw_dir = os.path.join(sys.argv[3], 'sub_' + str(subj_id))
work_dir = os.path.join(sys.argv[4], 'sub_' + str(subj_id))
out_dir = sys.argv[5].join(sys.argv[5], 'sub_' + str(subj_id))

print('Processing data for {}'.format(subj_id))

#Path to preprocessing requirements (atlases etc)
req_dir = '../atlases/'

###HUMAN CHOICE BITS BELOW###

#Confounds to remove - NOTE: MUST BE PRESENT IN fmriprep_confounds_fn
confounds_list = ['cosine00', 'cosine01','cosine02', 'cosine03', 'cosine04', 'cosine05', 'global_signal'] #Currenly low pass filters, CSF regression

#Full Width Half Maximum to smooth (mm)
fhwm_mm = 6

#Parcellation atlas
parc_fn = os.path.join(req_dir, 'aal116', 'aal116MNI_3mm.nii.gz')
parc_ids_fn = os.path.join(req_dir, 'aal116', 'aal116NodeNames.txt')

###HUMAN CHOICE BITS ABOVE###


#Imaging files
epi_fn = os.path.join(out_dir, 'func', 'sub-{}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(subj_id))
brain_mask_fn = os.path.join(out_dir, 'func', 'sub-{}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(subj_id))
dseg_fn = os.path.join(out_dir, 'anat', 'sub-{}_space-MNI152NLin2009cAsym_dseg.nii.gz'.format(subj_id))
lesion_mask_fn = glob.glob(os.path.join(raw_dir, 'anat', 'sub-{}_T1w_label-lesion_roi.nii'.format(subj_id)))
melodic_fn = os.path.join(work_dir, 'fmriprep_wf', 'single_subject_{}_wf', 'func_preproc_task_rest_wf', 'ica_aroma_wf', 'melodic', 'melodic_IC.nii.gz'.format(subj_id))

transform_fn = os.path.join(out_dir, 'anat', 'sub-{}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'.format(subj_id))

#Tabulated files
fmriprep_confounds_fn = os.path.join(out_dir, 'func', 'sub-{}_sub-tia001_task-rest_desc-confounds_regressors.tsv'.format(subj_id))
mixing_matrix_fn = os.path.join(out_dir, 'func', 'sub-{}_sub-tia001_'.format(subj_id))
noise_comps_fn = os.path.join(out_dir, 'func', 'sub-{}_task-rest_AROMAnoiseICs.csv'.format(subj_id))





#Cleaning
if lesion_mask_fn:
    lesion_mni_anat_fn = anat_apply_transforms(moving_fn, ref_fn, transform_fn, trans_type = 'label')
    lesion_mni_func_fn = anat2epi_ds(lesion_mni_anat_fn, brain_mask_fn)

    lesion_comps_fn =  epi_lesion_confound(lesion_mni_func_fn, melodic_fn, brain_mask_fn, thresh = 0.05)
    aroma_confounds_fn = epi_collect_ICA_confounds(mixing_matrix_fn, noise_comps_fn, lesion_comps_fn = lesion_comps_fn)
    selected_confounds_fn = epi_gen_confounds_matrix(fmriprep_confounds_fn, confounds_list)
    clean_fn = epi_clean(epi_fn, brain_mask_fn, selected_confounds_fn, aroma_confounds_fn, output_z = True)

    if len(clean_fn) > 1:
        smooth_fn = [epi_smooth(clean_single, mask_fn, fhwm_mm) for clean_single in clean_fn]
        aal_fn = [epi_parcellation(smooth_single, parc_fn, parc_ids_fn, atlas_name = 'aal') for smooth_single in smooth_fn]
        parc_overlap_fn = [parc_lesion_overlap(aal, lesion_mni_func_fn, parc_ids_fn, atlas_name = 'aal') for all in aal_fn]

    else:
        smooth_fn = epi_smooth(clean_fn, mask_fn,fhwm)
        aal_fn = epi_parcellation(smooth_fn, parc_fn, parc_ids_fn, atlas_name = 'aal')
        parc_overlap_fn = parc_lesion_overlap(parc_fn, lesion_mni_func_fn, parc_ids_fn, atlas_name = 'aal')

else:
    lesion_comps_fn =  epi_lesion_confound(lesion_fn, melodic_fn, brain_mask_fn, thresh = 0.05)
    aroma_confounds_fn = epi_collect_ICA_confounds(mixing_matrix_fn, noise_comps_fn, lesion_comps_fn  = None)
    selected_confounds_fn = epi_gen_confounds_matrix(fmriprep_confounds_fn, confounds_list)
    clean_fn = epi_clean(epi_fn, brain_mask_fn, selected_confounds_fn, aroma_confounds_fn, output_z = True)

    if len(clean_fn) > 1:
        smooth_fn = [epi_smooth(clean_single, mask_fn, fhwm_mm) for clean_single in clean_fn]
        aal_fn = [epi_parcellation(smooth_single, parc_fn, parc_ids_fn, atlas_name = 'aal') for smooth_single in smooth_fn]
    else:
        smooth_fn = epi_smooth(clean_fn, mask_fn,fhwm)
        aal_fn = epi_parcellation(smooth_fn, parc_fn, parc_ids_fn, atlas_name = 'aal')


#Lag calc

epi_size = np.load(epi_fn).size()
gm_mask_fn = anat_gm_mask(dseg_fn, epi_size)
lag_fn = epi_calc_lag(epi_fn, gm_mask_fn, brain_mask_fn)
