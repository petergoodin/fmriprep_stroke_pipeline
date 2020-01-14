import nibabel as nb
import numpy as np

msdl_sep = './msdl_rois.nii'

msdl_sep_hdr = nb.load(msdl_sep)

msdl_sep_data = msdl_sep_hdr.get_data()
x, y, z, nRois = msdl_sep_data.shape

msdl_join = []
# msdl_join = np.zeros(x, y, z)

for idx in range(0, nRois):
    print(idx)
    split = msdl_sep_data[:, :, :, idx].copy()
    split[split > 0.1] = idx
    print(int(split.max()))
    msdl_join.append(split)
    print(int(msdl_join[idx].max()))
msdl_join = np.array(msdl_join).T
msdl_join[msdl_join < 1] = 0
# print(msdl_join.T.shape)
# # msdl_join = msdl_join.sum(axis = 0)
# print(np.unique(msdl_join.T))


nb.Nifti1Image(msdl_join.T, header = msdl_sep_hdr.header, affine =  msdl_sep_hdr.affine).to_filename('./msdl_thresh.nii.gz')
