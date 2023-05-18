import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Set, Tuple, Any
from functools import partial
import torch
from torch.autograd import Variable

from transvae.trans_models import TransVAE
from transvae.rnn_models import RNNAttn
from transvae.data import vae_data_gen
from transvae.tvae_util import tokenizer
from scripts.parsers import latent_vector_parser


def search_illegal_tokens_and_length(smiles: str, model: Union[TransVAE, RNNAttn]) -> Tuple[Set, int]:
    tokens = tokenizer(smiles)
    illegal_tokens = set()
    char_dict = model.params['CHAR_DICT']
    for token in tokens:
        if token not in char_dict:
            illegal_tokens.add(token)
    return {'illegal_tokens': illegal_tokens, 'token_length': len(tokens)}

def check_validity(validation_summary: Dict[str, Any]):
    if len(validation_summary['illegal_tokens']) > 0:
        return False
    elif validation_summary['token_length'] > 127:
        return False
    return True

def calc_latent_vector(args):
    ### Load model
    ckpt_fn = args.model_ckpt
    if args.model_type == 'transvae':
        vae = TransVAE(load_fn=ckpt_fn)
    elif args.model_type == 'rnnattn':
        vae = RNNAttn(load_fn=ckpt_fn)
    
    # Read and check dataset compatibility
    all_smiles = pd.read_csv(args.mols, delimiter=args.delimiter)
    all_smiles['_validation_summary'] = all_smiles[args.smiles_col_name].map(partial(search_illegal_tokens_and_length, model=vae))
    all_smiles['smiles_valid'] = all_smiles['_validation_summary'].map(check_validity)
    
    valid_dataset = all_smiles[all_smiles.smiles_valid]
    valid_smiless = valid_dataset[[args.smiles_col_name]].to_numpy()

    ### Load data and prepare for iteration
    data = vae_data_gen(valid_smiless, props=None, char_dict=vae.params['CHAR_DICT'])
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
            dfs = []
            for i, batch_data in enumerate(data_iter):
                smiless = valid_smiless[i*args.batch_size:(i+1)*args.batch_size]
                mols_data = batch_data[:, :-1]
                props_data = batch_data[:,-1]
                if vae.use_gpu:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()
                
                src = Variable(mols_data).long()
                src_mask = (src != vae.pad_idx).unsqueeze(-2)
                latent_mem = vae.model.encoder.get_latent_vector(vae.model.src_embed(src), src_mask)
                dfs.append(pd.concat([pd.DataFrame(data={args.smiles_col_name: [s[0] for s in smiless]}), 
                                        pd.DataFrame(latent_mem.numpy(), columns=[f'{args.model_type}_{i}' for i in range(1, latent_mem.shape[1]+1)])],
                                        axis=1
                                        )
                )                    
                
    elif args.model_type == 'rnnattn':
        with torch.no_grad():
            with open(args.output, 'w') as output:
                output.write('smiles,'+','.join([f'rnnattn_{i}' for i in range(128)])+'\n')
                for i, batch_data in enumerate(data_iter):
                    smiless = all_smiles[i*args.batch_size:(i+1)*args.batch_size]
                    
    combined_df = pd.concat(dfs, ignore_index=True)
    merged = all_smiles[[args.smiles_col_name]].merge(combined_df, how='left', on=args.smiles_col_name)
    merged.to_csv(args.output, index=False)

if __name__ == '__main__':
    parser = latent_vector_parser()
    args = parser.parse_args()
    calc_latent_vector(args)
