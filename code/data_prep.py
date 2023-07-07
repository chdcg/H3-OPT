# -*- coding: utf-8 -*-
"""
1. delete not H chain structure
2. align all H chain structure
"""
from module.utils import *
from module.feature_extraction import feature_embed
import os
from schrodinger import structure
from schrodinger.structutils.structalign import StructAlign
from schrodinger.structutils.rmsd import superimpose
import argparse

def get_only_H_chain_st(pdbid,cdr_dir,st):
    prep_structure(st)
    all_seq = get_single_chain_resinfo(st)[0]
    anti_info = get_anti_info(cdr_dir,pdbid,all_seq)
    H_seq = get_VH_seq(anti_info)
    H_idx = get_seq_idx(H_seq,st)
    H_chain_st = delete_chain_seq(H_idx,st)
    return H_chain_st

def delete_chain_seq(res_idx,st):
    ct = st.copy()
    delete_atoms = []
    for res in ct.residue:
        if res.resnum not in res_idx:
            delete_atoms += res.getAtomIndices()
        else:
            pass
    ct.deleteAtoms(delete_atoms)
    prep_structure(ct)
    return ct


def prep_st(pdbid):
    print('Aligning pdbfiles to reference PDBID')
    aligned_st = read_single_st(args.input_structure_dir + REFERENCE_PDBID + '.pdb',REFERENCE_PDBID)
    raw_af2_st = read_single_st(args.input_structure_dir + pdbid + '.pdb',pdbid)
    af2_H_st = get_only_H_chain_st(pdbid,args.tmp_dir,raw_af2_st)
    af2_writer = structure.PDBWriter(new_af2_st_dir + '%s.pdb' % pdbid)
    a = StructAlign()
    a.alignStructure(aligned_st, af2_H_st)
    af2_writer.append(af2_H_st)

def prep_feats(pdbid):
    feature_embed(pdbid,args.tmp_dir,new_af2_st_dir,args.input_structure_dir,args.feature_dir)


if __name__ == '__main__':

    REFERENCE_PDBID = '1gig'
    WORKDIR = os.path.abspath(os.pardir) + '/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_structure_dir",  default=False,
        help="Path to input PDB file, please use PDB format files as inputs")

    parser.add_argument(
        "--feature_dir",  default=False,
        help="Path to output feature directory")

    parser.add_argument(
        "--tmp_dir",  default='../tmp/',
        help="Path to renumbering files")

    parser.add_argument(
        "--pdbname",  default=False,
        help="Pdbname of input PDB file")

    args = parser.parse_args()

    new_af2_st_dir = args.input_structure_dir + 'aligned_af2_st/'
    check_paths([new_af2_st_dir,args.feature_dir,args.tmp_dir])

     # only remain heavy chain and align all structures with reference PDB
    prep_st(args.pdbname)
    prep_feats(args.pdbname)


'''
USAGE:
$path_to_schrodinger/run data_prep.py --input_structure_dir ../af2_structure/ --feature_dir ../feats/ --pdbname 1a3r --tmp_dir  ../tmp/

'''