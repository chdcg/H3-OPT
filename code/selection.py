from module.utils import check_paths
from schrodinger import structure
from module.CBM import CBM
from module.TGM import TGM
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_structure_dir",  default=False,
        help="Path to input PDB file, please use PDB format files as inputs")

    parser.add_argument(
        "--output_structure_dir",  default=False,
        help="Path to output PDB directory")

    parser.add_argument(
        "--pdbname",  default=False,
        help="Pdbname of input PDB file")

    parser.add_argument(
        "--tmp_dir",  default='../tmp/',
        help="Path to renumbering files")

    parser.add_argument(
        "--template_dir",  default='../tmp/',
        help="Path to CDR-H3 template files")

    parser.add_argument(
        "--cutoff",  default=80,
        help="specify the cutoff of high confidence")
    args = parser.parse_args()

    print('=============Start CBM===============\n')
    cbm = CBM(args.input_structure_dir,args.pdbname,args.cutoff)
    check_paths([args.output_structure_dir])
    if not cbm.check_confidence(args.tmp_dir + '/'):
        print('=============Start TGM===============\n')

        tgm = TGM(args.template_dir)
        tgm.check_has_same_cdr3(args.pdbname,args.tmp_dir + '/',args.input_structure_dir,args.output_structure_dir)
    
'''
EXAMPLE:
$path_to_schrodinger/run selection.py --input_structure_dir ../af2_structure/ --tmp_dir ../tmp/ --pdbname 1a3r --output_structure_dir ../outpdb/ --template_dir ../template
'''