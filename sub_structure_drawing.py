import os
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole


def parse_fp_prior(fp_prior_f):
    df = pd.read_csv(fp_prior_f,sep='\t',index_col=0)
    # valid df : log2fc > 0
    valid_ser = df.loc[df.loc[:,'sign_Log2FC']>0,'rank_sum']
    return valid_ser.sort_values(ascending=False)
    

def _conv_img_to_base64(img,format="PNG"):
    im_file = BytesIO()
    img.save(im_file, format=format)
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    img_decoded = im_b64.decode()
    return img_decoded
    
    
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)),color='#FFFFFF')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height),color='#FFFFFF')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
    
    
def get_master_img(smiles:str,show_atom_label=True,show_image=False):
    # image size
    _whole_img_size = (600,600)
    _img_tmpl = "data:image/png;base64,"
    
    # Create a molecule from a SMILES string
    mol = Chem.MolFromSmiles(smiles)
    
    # Adding atom idx
    if show_atom_label:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        
    whole_img = Draw.MolToImage(mol,size=_whole_img_size)
    whole_img_str=_conv_img_to_base64(whole_img)
    whole_img_str=_img_tmpl+whole_img_str
    if show_image:
        return whole_img, whole_img_str
    else:
        return whole_img_str
    
    
    
def get_fing_imgs(smiles:str,
                  fing_prior_f:str=os.path.join(os.path.split(os.path.abspath(__file__))[0],'ecfp4_ranked_priority.txt'),
                  show_dup_fing:bool=True,
                  max_fing_n=5,
                  show_image:bool=False,
                 ):
    # image size
    _whole_img_size = (600,600)
    _fing_img_size = (100,100)
    _img_tmpl = "data:image/png;base64,"
    
    # Create a molecule from a SMILES string
    mol = Chem.MolFromSmiles(smiles)
    
    # Adding atom idx
    if show_dup_fing:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        
    # Generate the ECFP4 fingerprint for the molecule
    bi = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi)
    fp_on_bits = list(fp.GetOnBits())
    
    # Sorting fp by priority
    fp_prior_ser = parse_fp_prior(fing_prior_f)
    fp_prior_ser = fp_prior_ser.rename({i:int(i.split("_")[1]) for i in fp_prior_ser.index})
    fp_prior_ser = fp_prior_ser.loc[set(fp_prior_ser.index)&set(fp_on_bits)]
    fp_prior_ser = fp_prior_ser.sort_values(ascending=False)
    target_fp_l = list(fp_prior_ser.index)[:max_fing_n]

    # Drawing sub structure images
    sub_img_l = [] # [(FING_NAME,IMAGE,IMAGE_STR),...]
    for _curr_fp in target_fp_l:
        fing_idx = f'ECFP4:{_curr_fp:04d}'
        if show_dup_fing:
            _sub_img = Image.new('RGB',(0,0),color='#FFFFFF')
            for _atom_no_tup in bi[_curr_fp]:
                try:
                    _sub_bi_info = {_curr_fp:(_atom_no_tup,)}
                    _sub_img = get_concat_v(
                        _sub_img,
                        Draw.DrawMorganBit(mol,_curr_fp,_sub_bi_info,molSize=_fing_img_size))
                    _sub_img_str = _conv_img_to_base64(_sub_img)
                except:
                    continue
        else:
            try:
                _sub_img = Draw.DrawMorganBit(mol,_curr_fp,bi,molSize=_fing_img_size)
                _sub_img_str = _conv_img_to_base64(_sub_img)
            except:
                continue
            
            
        _sub_img_str = _img_tmpl+_sub_img_str
        sub_img_l.append(tuple([fing_idx,_sub_img,_sub_img_str]))
        
    sub_img_d = {}
    for _idx, fing_img_info in enumerate(sub_img_l):
        _img_title_var = f'candidate{_idx}_name'
        _img_var = f'candidate{_idx}_img'
        sub_img_d[_img_title_var] = fing_img_info[0]
        sub_img_d[_img_var] = fing_img_info[2]
        if show_image:
            _img_png_var = f'candidate{_idx}_img_png'
            sub_img_d[_img_png_var] = fing_img_info[1]
    
    return sub_img_d
