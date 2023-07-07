# -*- coding: utf-8 -*-
"""

"""
import os

import pandas as pd

from module.utils import *
import numpy as np
from schrodinger.structutils.measure import measure_dihedral_angle,measure_distance

def get_aa_type(seq,max_length = 160,
                aa_one_hot=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                             'W', 'Y', 'X']
                ):
    # return one hot encoding of sequence
    aa_type = np.zeros((max_length,21))
    for idx,s in enumerate(seq):
        aa_type[idx,aa_one_hot.index(s)] = 1
    return aa_type


def get_af2_all_pos_info(af2_st,H3_seq,H_idxs,max_length = 160):
    position = np.zeros((max_length,1))
    af2_H3_idxs = get_seq_idx(H3_seq,af2_st)
    for i in af2_H3_idxs:
        position[H_idxs.index(i)] = 1
    return position

def get_CA_xyzs(af2_st,max_length = 160):
    xyzs = np.zeros((max_length,3))
    for idx,res in enumerate(af2_st.residue):
        ca = res.getAlphaCarbon()
        xyz = np.array(ca.xyz)
        xyzs[idx] = xyz
    return  xyzs

def get_dihedral_angles(af2_st,max_length=160,\
            diheral_angle = ['Omega','Phi', 'Psi']):
    dihedral_angles = np.zeros((max_length,9))
    for idx,res in enumerate(af2_st.residue):
        for angle_idx,angle_type in enumerate(diheral_angle):
            try:
                atoms = res.getDihedralAtoms(angle_type)
                angle = measure_dihedral_angle(atoms[0],atoms[1],atoms[2],atoms[-1])
                dihedral_angles[idx,2*angle_idx] = np.sin(angle/180)
                dihedral_angles[idx,2*angle_idx + 1] = np.cos(angle/180)
                dihedral_angles[idx,6 + angle_idx] = 1
            except Exception as e:
                print(e)
                dihedral_angles[idx,2*angle_idx] = 0
                dihedral_angles[idx,2*angle_idx + 1] = 0
                dihedral_angles[idx,6 + angle_idx] = 0
    return dihedral_angles

def get_pair_feat(af2_st,aa_type,min_dis = 3.25, cutoff = 1.25,MAX_SEQ_LENGTH=160):
    pairs = np.zeros((MAX_SEQ_LENGTH,MAX_SEQ_LENGTH,39))
    a = aa_type.reshape(MAX_SEQ_LENGTH,1,21)
    b = aa_type.reshape(1,MAX_SEQ_LENGTH,21)
    aa_feats = a + b
    for idx,res in enumerate(af2_st.residue):
        mol_ca = res.getAlphaCarbon()
        for ref_idx,ref_res in enumerate(af2_st.residue):
            ref_ca = ref_res.getAlphaCarbon()
            dis = measure_distance(mol_ca,ref_ca)
            if dis >= 50.75:
                pairs[idx, ref_idx, 38] = 1
            elif dis <= 3.25:
                pairs[idx, ref_idx, 0] = 1
            else:
                pairs[idx, ref_idx, int((dis - min_dis)//cutoff) ] = 1
    new_pair_feats = np.concatenate((pairs,aa_feats),axis=-1)
    return new_pair_feats


def get_raw_all_res_xyz(raw_st,raw_H3_idx):
    xyzs = np.zeros((44, 3))

    index = 0
    for res in raw_st.residue:
        if res.resnum in raw_H3_idx:
            xyzs[index,:] = np.array(res.getAlphaCarbon().xyz)
            index += 1
    return xyzs


def feature_embed(pdbname,cdr_dir,af2_st_dir, raw_af2_st_dir, feat_dir):

    all_res_feats = {}

    '''pdbids of aligned_af2_st_dir has been filtered by missing loop pdbids'''
    af2_st = read_single_st(af2_st_dir + '%s.pdb' % pdbname, pdbname)
    raw_af2_st = read_single_st(raw_af2_st_dir+ '%s.pdb' % pdbname, pdbname)
    af2_seq = get_single_chain_resinfo(raw_af2_st)[0]
    anti_info = get_anti_info(cdr_dir, pdbname,af2_seq)
    H3_seq = get_cdr_seq(anti_info, 3)
    H_seq = get_VH_seq(anti_info)
    L_seq = get_VL_seq(anti_info)
    H_idxs = get_seq_idx(H_seq, af2_st)
    # # feature extraction
    pred_pos_encoding = get_af2_all_pos_info(af2_st, H3_seq, H_idxs)  # Nres *1
    aa_feat = get_aa_type(H_seq)  # Nres*3
    xyz_feat = get_CA_xyzs(af2_st)  # Nres *3
    dihedral_feat = get_dihedral_angles(af2_st)  # Nres *6
    res_feat = np.concatenate((aa_feat, pred_pos_encoding, xyz_feat, dihedral_feat), axis=1)
    # pair feat
    pair_feat = get_pair_feat(af2_st,aa_feat) # Nres * Nres *39
    np.save(feat_dir + '%s_pair.npy'%pdbname,pair_feat)
    all_res_feats['H_seq'] = H_seq
    all_res_feats['L_seq'] = L_seq
    all_res_feats['all_res'] = res_feat
    pd.to_pickle(all_res_feats, feat_dir + '%s_res.pkl'%pdbname)






