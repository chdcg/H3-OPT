from schrodinger import structure
import argparse,os
from module.utils import read_single_st,final_refine_structure,read_antibody_Renum,get_seq_idx,get_cdr_seq
from schrodinger.protein.captermini import cap_termini

import numpy as np
import pandas as pd


def cap_terminals(st):
    capped_st = cap_termini(st)
    return capped_st

def modify_af2_by_predict_xyz(pdbid,all_pdb_dir,cdr_dir,predicted_xyzs,new_st_dir):
    af2_0_st = read_single_st(all_pdb_dir + '%s.pdb'%pdbid , pdbid)
    anti_cdr_info = read_antibody_Renum(cdr_dir, pdbid)
    H3_seq = get_cdr_seq(anti_cdr_info[pdbid.upper()],3)
    af2_H3_idx = get_seq_idx(H3_seq, af2_0_st)
    actual_H3_num = len(H3_seq)
    predicted_xyz = list(np.array(predicted_xyzs.loc[pdbid]).reshape(44,3)[:actual_H3_num,:])
    change_st_by_xyz(af2_0_st,af2_H3_idx,predicted_xyz,pdbid,new_st_dir)

def change_af2_all_res_xyz(af2_st,af2_H3_idx,pred_xyzs):
    af2_desired_resnum = af2_H3_idx
    frozen_atoms = []
    idx = 0
    for res in af2_st.residue:
        if res.resnum in af2_desired_resnum:
            ca = res.getAlphaCarbon()
            ca.xyz = pred_xyzs[idx]
            frozen_atoms.append(ca.index)
            idx += 1
        elif res.resnum in [af2_H3_idx[0],af2_H3_idx[-1]]:#freeze first and last ca
            ca = res.getAlphaCarbon()
            frozen_atoms.append(ca.index)
    return frozen_atoms

def change_st_by_xyz(af2_0_st,af2_H3_idx,raw_five_xyzs,pdbid,new_st_dir):
    try:
        capped_st = cap_terminals(af2_0_st)
    except:
        capped_st = af2_0_st

    frozen_atoms = change_af2_all_res_xyz(capped_st, af2_H3_idx, raw_five_xyzs)
    ct = final_refine_structure(af2_H3_idx,capped_st,frozen_atoms)
    ct = final_refine_structure(af2_H3_idx,ct,constraint_atoms=frozen_atoms)
    writer = structure.PDBWriter(new_st_dir + '%s.pdb' % pdbid)
    writer.append(ct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    WORKDIR = os.path.abspath(os.path.join('..')) + '/'

    parser.add_argument(
        "--input_structure_dir",  default=False,
        help="Path to input PDB file, please use PDB format files as inputs")

    parser.add_argument(
        "--tmp_dir",  default='../tmp/',
        help="Path to renumbering files")
    
    parser.add_argument(
        "--output_structure_dir",  default=False,
        help="Path to output PDB directory")
    
    parser.add_argument(
        "--pdbname",  default=False,
        help="Pdbname of input PDB file")

    parser.add_argument(
        "--pred_csv",  default=False,
        help="filename of predicted cordinate files")    

    args = parser.parse_args()

    predicted_xyzs = pd.read_csv(args.pred_csv,index_col=0)
    modify_af2_by_predict_xyz(args.pdbname,
        os.path.join(args.input_structure_dir,'aligned_af2_st/') ,args.tmp_dir,predicted_xyzs,args.output_structure_dir)

'''
EXAMPLE
$path_to_schrodinger/run structure_generation.py --input_structure_dir ../af2_structure/ --tmp_dir ../tmp/ --output_structure_dir ../outpdb/ --pdbname 5u3p --pred_csv ../outpdb/test1.csv
'''
