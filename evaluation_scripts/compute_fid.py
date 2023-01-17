from data.mri_data import SelectiveSliceData_Val
from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.fid.fid_metric import FIDMetric
from data_loaders.prepare_data import create_data_loaders, DataTransform
from data_loaders.prepare_data_ls import create_data_loaders_ls
from wrappers.our_gen_wrapper import load_best_gan
from torch.utils.data import DataLoader
import numpy as np

def get_fid(args, G, ref_loader, cond_loader, max=1000000):
    print("GETTING VGG EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    fid_metric = FIDMetric(gan=G,
                           ref_loader=ref_loader,
                           loader=cond_loader,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=args,
                           max_num=max)

    return fid_metric.get_fid()