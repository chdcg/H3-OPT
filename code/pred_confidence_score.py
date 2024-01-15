from model.h3_opt_score import AFNet,embed_seq
import torch,esm,argparse,os
from model.AF2_models import RowAttentionWithPairBias, OuterProductMean
from model.triangular_multiplicative_update_init import *
import pandas as pd
def load_data(feature_dir,pdbid):
    res_feats = torch.tensor(pd.read_pickle(feature_dir + pdbid + '_res.pkl')['all_res'],dtype=torch.float,requires_grad=False).detach()
    pair_feats = torch.tensor(np.load(feature_dir+ '%s_pair.npy' % (pdbid)),dtype=torch.float,requires_grad=False)
    res_feats = res_feats.unsqueeze(0)
    pair_feats = pair_feats.unsqueeze(0)
    return res_feats,pair_feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_name = 'plddt.pth'
    WORKDIR = os.path.abspath(os.path.join('..')) + '/'

    parser.add_argument(
        "--feature_dir", type=str, default=os.path.join(WORKDIR, 'feats/'),
        help="Path to feature files")

    parser.add_argument(
        "--model_dir", type=str, default=os.path.join(WORKDIR, 'models/'),
        help="Path to model directory ")

    parser.add_argument(
        "--out_dir", type=str, default=os.path.join(WORKDIR, 'analysis/'),
        help="Path to output PDB directory")

    parser.add_argument(
        "--out_name", type=str, default='test',
        help="filename of predicted cordinate files")

    parser.add_argument(
        "--pdbname", default=False,
        help="Pdbname of input PDB file")

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model_dir + model_name)
    model.to(device)
    embeddings = embed_seq(args.feature_dir,args.pdbname,device)
    res_feats,pair_feats = load_data(args.feature_dir,args.pdbname)
    plddt = model(res_feats.to(device),pair_feats.to(device),embeddings).cpu()[0][0]
    print("The confidence score of this loop is {:.3f}".format(plddt.detach().numpy()[0]))

'''
USAGE:
python pred_confidence_score.py --feature_dir ../feats/ --model_dir ../model/ --out_dir ../outpdb/ --out_name test1 --pdbname 1a3r

'''
