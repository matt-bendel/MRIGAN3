from data.mri_data import SelectiveSliceData_Val
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from data_loaders.prepare_data import create_data_loaders, DataTransform
from data_loaders.prepare_data_ls import create_data_loaders_ls
from wrappers.our_gen_wrapper import load_best_gan
from torch.utils.data import DataLoader
import numpy as np

def get_lang_data_loaders(args):
    data = SelectiveSliceData_Val(
        root=args.data_path / 'small_T2_test',
        transform=DataTransform(args, test=True),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=6,
        restrict_size=False,
        big_test=True
    )

    loader = DataLoader(
        dataset=data,
        batch_size=4,
        num_workers=16,
        pin_memory=True
    )

    return None, loader

def get_cfid(args, G, loader=False, ref_loader=False, num_samps=32, langevin=False):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = VGG16Embedding(parallel=True)

    cfid_metric = CFIDMetric(gan=G,
                             loader=loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args,
                             ref_loader=ref_loader,
                             num_samps=num_samps)

    cfids = cfid_metric.get_cfid_torch_pinv()
    del cfid_metric
    print(f'CFID: {cfids}')

    return np.mean(cfids)
