from evaluation_scripts.cfid.embeddings import InceptionEmbedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from data_loaders.prepare_data import create_data_loaders
from wrappers.our_gen_wrapper import load_best_gan


def get_cfid(args, G):
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)

    print("GETTING DATA LOADERS")
    _, loader = create_data_loaders(args, val_only=True, big_test=True)

    cfid_metric = CFIDMetric(gan=G,
                             loader=loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=args)

    print(f'CFID: {cfid_metric.get_cfid_torch()}')
