from utils import read_antibody_Renum, get_cdr_seq,read_single_st,get_seq_idx,final_refine_structure
import pandas as pd
from schrodinger.structutils.build import add_hydrogens, delete_hydrogens
from schrodinger.application.prepwizard import do_any_residues_have_missing_side_chains
from schrodinger import structure
from schrodinger.structutils.rmsd import superimpose



BAD_TEMPLATE = ['1iga','6jfi'] #wrong templates

def extract_template_cdr3(res_info,pdbid,cdr_dir,temp_dir):
    print('Found same CDR3 in %s'%res_info)
    temp_st = read_single_st(temp_dir + res_info + '.pdb',res_info)
    anti_cdr_info = read_antibody_Renum(cdr_dir, pdbid)
    H3_seq = get_cdr_seq(anti_cdr_info[pdbid.upper()],3)
    H3_idx = get_seq_idx(H3_seq, temp_st)
    delete_not_cdr3_atoms(temp_st,H3_idx)
    return temp_st

def delete_not_cdr3_atoms(st, H3_idxs):
    delete_atoms = []
    first_cdr3_resnum,last_cdr3_resnum = H3_idxs[0],H3_idxs[-1]
    fr_res = list(range(first_cdr3_resnum-2,first_cdr3_resnum)) + \
             list(range(last_cdr3_resnum+1, last_cdr3_resnum+3))
    remain_resnum = list(set(H3_idxs + fr_res))
    for res in st.residue:
        if res.resnum not in remain_resnum:
            delete_atoms += res.getAtomIndices()
    st.deleteAtoms(delete_atoms)

def change_af2_xyzs(temp_cdr3_st,af2_st,af2_H3_idxs):
    for atom in af2_st.atom:
        if atom.resnum in af2_H3_idxs:
            index = af2_H3_idxs.index(atom.resnum)
            for temp_atom in temp_cdr3_st.atom:
                # print(atom.pdbname,temp_atom.pdbname)
                if temp_atom.pdbname == atom.pdbname and temp_atom.resnum == index :
                    atom.xyz = temp_atom.xyz

def align_temp_cdr3_with_af2(temp_st, af2_st):
    aligned_temp_atoms = []
    aligned_af2_atoms = []
    H3_seq = ''
    h3_idx = 0
    for idx ,res in enumerate(temp_st.residue):
        if idx < 2:
            aligned_temp_atoms.append(res.getAlphaCarbon().index)
            aligned_temp_atoms.append(res.getCarbonylCarbon().index)
        elif idx >= len(temp_st.residue) -2:
            aligned_temp_atoms.append(res.getAlphaCarbon().index)
            aligned_temp_atoms.append(res.getBackboneNitrogen().index)
        else:
            res.resnum = h3_idx  # renumberring template cdr3 structure
            H3_seq += res.getCode()
            h3_idx += 1
    
    H3_idx = get_seq_idx(H3_seq, af2_st)
    for res in af2_st.residue:
        if res.resnum < H3_idx[0] and res.resnum >= H3_idx[0]-2:
            aligned_af2_atoms.append(res.getAlphaCarbon().index)
            aligned_af2_atoms.append(res.getCarbonylCarbon().index)
        elif res.resnum > H3_idx[-1] and res.resnum <= H3_idx[-1] + 2:
            aligned_af2_atoms.append(res.getAlphaCarbon().index)
            aligned_af2_atoms.append(res.getBackboneNitrogen().index)
    superimpose(af2_st,aligned_af2_atoms,temp_st,aligned_temp_atoms)
    return H3_idx

def modify_af2_st_by_template(temp_cdr3_st, af2_pdb_dir, pdbid, outdir):
    af2_st = read_single_st(af2_pdb_dir + '%s.pdb'%pdbid,pdbid)
    do_any_residues_have_missing_side_chains(temp_cdr3_st)
    af2_H3_idxs = align_temp_cdr3_with_af2(temp_cdr3_st,af2_st)
    # tmp = af2_st.copy()
    change_af2_xyzs(temp_cdr3_st,af2_st,af2_H3_idxs)
    delete_hydrogens(af2_st)
    add_hydrogens(af2_st)
    frozen_atoms = [atom.index for atom in af2_st.atom if atom.resnum in af2_H3_idxs]
    ct = final_refine_structure(af2_H3_idxs, af2_st,frozen_atoms=frozen_atoms)
    print('Finish grafting H3 loop from template!')
    writer = structure.PDBWriter(outdir + '%s.pdb' % pdbid)
    writer.append(ct)


class TGM(object):
    def __init__(self, path_to_template):
        self.path_to_template = path_to_template
        self.temp_cdr3_seq = pd.read_pickle(self.path_to_template + 'all_H3_seqs.pkl')
    
    def check_has_same_cdr3(self,pdbid,cdr_dir,input_structure_dir,output_structure_dir):
        tmp = False
        anti_cdr_info = read_antibody_Renum(cdr_dir, pdbid)
        H3_seq = get_cdr_seq(anti_cdr_info[pdbid.upper()],3)
        for ref_info, seqs in self.temp_cdr3_seq.items():
            if H3_seq in seqs and pdbid not in ref_info:
                if ref_info.split('_')[0] not in BAD_TEMPLATE:
                    tmp = ref_info
                    temp_cdr3_st = extract_template_cdr3(tmp,pdbid,cdr_dir,'../template/cleaned_pdb/')
                    modify_af2_st_by_template(temp_cdr3_st,input_structure_dir, pdbid,output_structure_dir)
        if not tmp:
            print('Failed to find template loop, please try DLBM')