import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from transvae.trans_models import TransVAE
from transvae.rnn_models import RNNAttn
from transvae.data import vae_data_gen
from scripts.parsers import latent_vector_parser


def calc_latent_vector(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model_type == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model_type == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    
    all_smiles = pd.read_csv(args.mols).to_numpy()
    
    ### Load data and prepare for iteration
    data = vae_data_gen(all_smiles, props=None, char_dict=vae.params['CHAR_DICT'])
    data_iter = torch.utils.data.DataLoader(data,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=0,
                                            pin_memory=False, drop_last=False)
            
    ### Prepare save folder path
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ### SMILES to latent vector
    vae.model.eval()
    if args.model_type == 'transvae':
        with torch.no_grad():
            with open(args.output, 'w') as output:
                output.write('smiles,'+','.join([f'transvae_{i}' for i in range(128)])+'\n')
                for i, batch_data in enumerate(data_iter):
                    smiless = all_smiles[i*args.batch_size:(i+1)*args.batch_size]
                    mols_data = batch_data[:, :-1]
                    props_data = batch_data[:,-1]
                    if vae.use_gpu:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()
                    
                    src = Variable(mols_data).long()
                    src_mask = (src != vae.pad_idx).unsqueeze(-2)
                    latent_mem = vae.model.encoder.get_latent_vector(vae.model.src_embed(src), src_mask)
                    for smiles, row_values in zip(smiless, latent_mem):
                        output.write(f'{smiles[0]},')
                        output.write(",".join(str(v) for v in row_values.tolist()))
                        output.write("\n")
                    
    elif args.model_type == 'rnnattn':
        with torch.no_grad():
            pass

if __name__ == '__main__':
    parser = latent_vector_parser()
    args = parser.parse_args()
    calc_latent_vector(args)
