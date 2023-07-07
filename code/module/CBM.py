import os
from schrodinger import structure
from .utils import read_single_st,get_anti_info,get_seq_idx,get_cdr_seq,\
    get_single_chain_resinfo,prep_structure
import numpy as np


class CBM(object):
    def __init__(self, path_to_pdbid,pdbid,cutoff):
        super(CBM, self).__init__()
        self.path_to_pdbid = path_to_pdbid
        self.pdbid = pdbid
        self.cutoff = cutoff
        self.st = read_single_st(self.path_to_pdbid +'/'+ self.pdbid+'.pdb', pdbid )

    def get_H3_average_bfactor(self,cdr_dir):
        prep_structure(self.st)
        assert len(self.st.chain) == 1 # please only remain VH of 
        all_seq = get_single_chain_resinfo(self.st)[0]
        af2_anti_info = get_anti_info(cdr_dir, self.pdbid,all_seq)
        af2_H3_idxs = get_seq_idx(get_cdr_seq(af2_anti_info, 3), self.st)
        confidence = self.get_H3_atom_bfactor(af2_H3_idxs)
        return np.mean(confidence)


    def get_H3_atom_bfactor(self,H3_idx_list):
        cdr_ca_indices = []
        atom_bfactors = []
        for res in self.st.residue:
            if res.resnum in H3_idx_list:
                atom_indices = res.getAlphaCarbon().index
                cdr_ca_indices.append( atom_indices )
        for atom_idx in cdr_ca_indices:
            atom = self.st.atom[atom_idx]
            atom_bfactors.append(atom.temperature_factor)
        return atom_bfactors

    def check_confidence(self,cdr_dir):
        confidence = self.get_H3_average_bfactor(cdr_dir)
        if confidence >= self.cutoff:

            print('Input CDR-H3 loop has high confidence score : %s '%self.pdbid)
            return True
        else:
            print('Input CDR-H3 loop has low confidence score : %s , try TGM...'%self.pdbid)
            return False
