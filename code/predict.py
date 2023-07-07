import torch,argparse,os
import pandas as pd
import numpy as np
from model.h3_opt import AFNet_ft,AFNet

def check_path(path):
    # create path if not exist
    if not os.path.exists(path):
        os.mkdir(path)

def load_data(feature_dir,pdbid):
    res_feats = torch.tensor(pd.read_pickle(feature_dir + pdbid + '_res.pkl')['all_res'],dtype=torch.float,requires_grad=False).detach()
    pair_feats = torch.tensor(np.load(feature_dir+ '%s_pair.npy' % (pdbid)),dtype=torch.float,requires_grad=False)
    res_feats = res_feats.unsqueeze(0)
    pair_feats = pair_feats.unsqueeze(0)
    return res_feats,pair_feats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_name = 'best_wt.pth'
    WORKDIR = os.path.abspath(os.path.join('..')) + '/'

    parser.add_argument(
        "--feature_dir" ,type = str, default= os.path.join( WORKDIR, 'feats/'),
        help="Path to feature files")

    parser.add_argument(
        "--model_dir" , type = str,default= os.path.join( WORKDIR,  'models/'),
        help="Path to model directory ")

    parser.add_argument(
        "--out_dir",type = str,  default= os.path.join( WORKDIR,  'analysis/'),
        help="Path to output PDB directory")
    
    parser.add_argument(
        "--out_name",type = str,  default= 'test',
        help="filename of predicted cordinate files")    
    
    parser.add_argument(
        "--pdbname",  default=False,
        help="Pdbname of input PDB file")

    args = parser.parse_args()
    check_path(args.out_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AFNet_ft(35)
    model.to(device)
    model.load_state_dict(torch.load(args.model_dir + model_name))
    model.eval()
    res_feats,pair_feats = load_data(args.feature_dir,args.pdbname)
    pred = model(res_feats.to(device),pair_feats.to(device),args.pdbname,args.feature_dir)
    df = pd.DataFrame(pred.squeeze(dim=1).cpu().detach().numpy(), index = [args.pdbname])
    df.to_csv(args.out_dir + args.out_name +'.csv')
'''
USAGE:
python predict.py --feature_dir ../feats/ --model_dir ../models/ --out_dir ../outpdb/ --out_name test1 --pdbname 1a3r

'''
