# -*- coding: utf-8 -*-
"""

"""
import subprocess,decimal,os
import pandas as pd
from schrodinger import structure
from schrodinger.structutils.rmsd import superimpose
import schrodinger.structutils.minimize as minimize
from schrodinger.structutils.rmsd import calculate_in_place_rmsd

PATH_TO_ABRSA = '/home/user/software/AbRSA'

def linux_cmd(cmd):
    p =subprocess.Popen(cmd,shell =True)
    p.wait()

def check_paths(paths):
    for i in paths:
        if not os.path.exists(i):
            os.makedirs(i)

def read_single_st(filename,pdbid):
    st = structure.StructureReader(filename).next()
    st.title = pdbid
    return st

def final_refine_structure(af2_H3_idx,st,frozen_atoms=[],constraint_atoms=[]):
    '''
    1. cap terminals for structure(delete not H chain)
    2. minimize not framework region atoms and forzen atoms
    '''
    frozen_atom_list = []
    if frozen_atoms:
        frozen_atom_list = frozen_atoms
    for res in st.residue:
        if res.resnum not in af2_H3_idx:
            frozen_atom_list += res.getAtomIndices()
    # print(frozen_atom_list)
    ct = minimizeLocalStructure(st,frozen_atom_index_list=frozen_atom_list,constraint_atom_number_list=constraint_atoms)
    return ct

def minimizeLocalStructure(st,frozen_atom_index_list=[],constraint_atom_number_list=[],force_constant = 10, dielectric = decimal.Decimal(78.3553)):
    '''
    atom in frozen atom list is frozen
    atom in constraint atom list is constraint with a force constant
    '''
    ct = st.copy()
    min = minimize.Minimizer(struct=ct,ffld_version=14,dielectric_constant=dielectric) # opls_2005
    min.deleteAllRestraints()
    for atom in ct.atom:
        if atom.index in constraint_atom_number_list:
            min.addPosRestraint(atom.index,force_constant)
        if atom.index in frozen_atom_index_list:
            min.addPosFrozen(atom.index)
    min.minimize()
    while min.min_converged != 1:
        min.minimize()
    minimized_ct = min.getStructure()
    return minimized_ct

def read_antibody_Renum(path,pdbid):
    with open(path+'%s_renum'%pdbid) as f:
        text = f.readlines()
        idx_list = []
        for idx, line in enumerate(text):
            if '>' in line:
                idx_list.append(idx)
    antibody_info = {}
    for index, idx in enumerate(idx_list):
        if idx != idx_list[-1]:
            tmp = [seq.strip() for seq in text[idx+3:idx_list[index+1]-1]]
        else:
            tmp = [seq.strip() for seq in text[idx+3:]]
        antibody_info[text[idx].split()[1]] = tmp
    return antibody_info

def get_VH_seq(lines):
    cdr_res_in_H = ''
    for line in lines:
        line = line.strip()
        if "H_" in line:
            cdr_res_in_H += line.split(":")[1].strip()
    return cdr_res_in_H

def get_VL_seq(lines):
    cdr_res_in_L = ''
    for line in lines:
        line = line.strip()
        if "L_" in line:
            cdr_res_in_L += line.split(":")[1].strip()
    return cdr_res_in_L

def get_anti_info(cdr_dir,pdbid,seq):
    with open(cdr_dir + '%s.fasta' % pdbid, 'w') as f:
        f.write('> %s\n' % (pdbid))
        f.write('%s\n' %seq)
    linux_cmd(PATH_TO_ABRSA + ' -i %s > %s' % (cdr_dir + pdbid + '.fasta', cdr_dir + pdbid + '_renum'))
    anti_cdr_info = read_antibody_Renum(cdr_dir, pdbid)
    anti_info = anti_cdr_info[pdbid.upper()]
    return anti_info

def prep_structure(st):
    ### merge all chains and renumberring all residue
    for idx, res in enumerate(st.residue):
        res.resnum = idx
        res.chain = 'A'


def get_single_chain_resinfo(st):
    seq = ''
    idx = []
    for res in st.residue:
        seq += res.getCode()
        idx.append(res.resnum)
    return seq,idx


def get_seq_idx(res,st):
    if res:
        seq,idx = get_single_chain_resinfo(st)
        index = seq.find(res)
        cdr_start_index = idx[index]
        length = len(res)
        index_end = index + length
        cdr_end_index = idx[index_end-1]
        res_index_list = list(range(cdr_start_index,cdr_end_index+1))
        return res_index_list
    else:
        return []

def get_cdr_seq(renum_info,cdr):
    for line in renum_info:
        if 'H_CDR%s'%cdr in line:
            seq = line.split(':')[1].strip()
    return seq

def filter_loss_atoms(atom_list1, st1, st2):
    # filter missing atoms
    loss_atoms = []
    atom_list2 = []
    for atom1_idx in atom_list1:
        atom1 = st1.atom[atom1_idx]
        check_loss_atom = True
        for atom2 in st2.atom:
            if atom1.pdbname.strip() == atom2.pdbname.strip() and atom1.resnum == atom2.resnum:
                atom_list2.append(atom2.index)
                check_loss_atom = False
                break
        if check_loss_atom:
            loss_atoms.append(atom1.index)
    if loss_atoms:
        for i in loss_atoms:
            atom_list1.remove(i)
    return atom_list1,atom_list2

def get_common_cdr_ca_list(st1,st2,cdr_residue_idx1):
    #generate cdr atom list dict of all models
    atom_list1 = []
    for  res1 in st1.residue:
        if res1.resnum in cdr_residue_idx1:
            try:
                res1_atoms = [atom for atom in res1.atom if atom.pdbname.strip() == 'CA']
                res1_atom_idx = [atom.index for atom in res1_atoms]
                atom_list1 += res1_atom_idx
            except Exception as e:
                print(e)
    atom_list1, atom_list2 = filter_loss_atoms(atom_list1, st1, st2)
    return [atom_list1, atom_list2]


