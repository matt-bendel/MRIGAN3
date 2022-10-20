from data.mri_data import SelectiveSliceData_Val
from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.fid.fid_metric import FIDMetric
from data_loaders.prepare_data import create_data_loaders, DataTransform
from data_loaders.prepare_data_ls import create_data_loaders_ls
from wrappers.our_gen_wrapper import load_best_gan
from torch.utils.data import DataLoader
import numpy as np

def get_fid(args, G, ref_loader, cond_loader):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")

    fid_metric = FIDMetric(gan=G,
                           ref_loader=train_loader,
                           loader=cond_loader,
                           image_embedding=inception_embedding,
                           condition_embedding=inception_embedding,
                           cuda=True,
                           args=args)

    fid_metric.get_fid()