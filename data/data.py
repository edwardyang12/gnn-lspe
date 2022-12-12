"""
    File to load dataset based on user control from main file
"""
from data.ogb_mol import OGBMOLDataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC' or DATASET_NAME == 'ZINC-full':
        from data.molecules import MoleculeDataset
        return MoleculeDataset(DATASET_NAME)

    # handling for (AQSOL) molecule dataset
    if DATASET_NAME == 'AQSOL':
        from data.aqsol import MoleculeDataset
        return MoleculeDataset(DATASET_NAME)
    
    # handling for MOLPCBA and MOLTOX21 dataset
    if DATASET_NAME in ['OGBG-MOLPCBA', 'OGBG-MOLTOX21']:
        return OGBMOLDataset(DATASET_NAME)