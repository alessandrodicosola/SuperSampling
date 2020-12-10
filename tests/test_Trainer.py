import unittest
from functools import partial
from unittest import TestCase
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from base.Trainer import Trainer
from datasets.ASDNDataset import ASDNDataset, collate_fn
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation


class TestTrainer(TestCase):

    def test_fit(self):

        lfr = LaplacianFrequencyRepresentation(1,2,11)
        collate_fn_lfr = partial(collate_fn,lfr = lfr)

        val_dataset = ASDNDataset("DIV2K_valid_HR", 24, lfr)
        val_loader = DataLoader(val_dataset, 64, False, num_workers=4, collate_fn=collate_fn_lfr)
        train_dataset = ASDNDataset("DIV2K_train_HR", 24,lfr)
        train_loader  = DataLoader(train_dataset,32,True,num_workers=4,collate_fn=collate_fn_lfr)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        asdn = ASDN(3,lfr,n_dab=3,n_intra_layers=3,out_compressed_channels=32,out_channels_dab=8,intra_layer_output_features=8).to(device)

        adam = Adam(asdn.parameters(),1e-3,betas=(0.99,0.999),eps=1e-8)
        trainer = Trainer("test",asdn,adam,MSELoss().to(device),device=device)
        trainer.fit(train_loader,val_loader,1)

if __name__ == "__main__":
    unittest.main()