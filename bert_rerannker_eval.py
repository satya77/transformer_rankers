from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_scisumm_ranked
from transformer_rankers.eval import results_analyses_tools

from transformers import BertTokenizer, BertForSequenceClassification
from sacred.observers import FileStorageObserver
from sacred import Experiment
import numpy as np
import torch
import pandas as pd
import argparse
import logging
import sys

ex = Experiment('BERT-ranker experiment')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


@ex.main
def run_experiment(args):
    args.run_id = str(ex.current_run._id)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train, valid, test = preprocess_scisumm_ranked.transform_to_dfs(
        args.path_to_ranked_file,args.path_to_ranked_test,args.path_to_ranked_dev)


    # Choose the negative candidate sampler
    ns_train=None
    ns_val=None

    # Create the loaders for the datasets, with the respective negative samplers
    dataloader = dataset.QueryDocumentDataLoader(train, valid, test,
                                                 tokenizer, ns_train, ns_val,
                                                 'classification', args.val_batch_size,
                                                 args.val_batch_size, 512,
                                                 0, args.data_folder +   "/scisumm_ranked")
    with_ranked_list=True
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders(with_ranked_list)

    # Instantiate transformer model to be used
    model = BertForSequenceClassification.from_pretrained('bert-base-cased')
    model.resize_token_embeddings(len(dataloader.tokenizer))
    e = torch.load(args.model_dir)
    model.load_state_dict(e)


    model.eval()
    # Instantiate trainer that handles fitting.
    trainer = transformer_trainer.TransformerTrainer(model, train_loader, val_loader, test_loader,
                                                     0, "classification", tokenizer,
                                                     False, 0,
                                                     0 ,0, 0)
    # Predict for test
    logging.info("Predicting")
    preds, labels, doc_ids, all_queries, preds_without_acc = trainer.test()
    # res = results_analyses_tools.evaluate_and_aggregate(preds, labels, ['R_10@1',
    #                                                                     'R_10@2',
    #                                                                     'R_10@5',
    #                                                                     'R_2@1',
    #                                                                     'accuracy_0.3',
    #                                                                     'accuracy_0.3_upto_1',
    #                                                                     'precision_0.3',
    #                                                                     'recall_0.3',
    #                                                                     'f_score_0.3',
    #                                                                     'accuracy_0.4',
    #                                                                     'accuracy_0.4_upto_1',
    #                                                                     'precision_0.4',
    #                                                                     'recall_0.4',
    #                                                                     'f_score_0.4',
    #                                                                     'accuracy_0.5',
    #                                                                     'accuracy_0.5_upto_1',
    #                                                                     'precision_0.5',
    #                                                                     'recall_0.5',
    #                                                                     'f_score_0.5'
    #                                                                     ])
    # for metric, v in res.items():
    #     logging.info("Test {} : {:4f}".format(metric, v))

    # # Saving predictions and labels to a file
    # max_preds_column = max([len(l) for l in preds])
    # preds_df = pd.DataFrame(preds, columns=["prediction_" + str(i) for i in range(max_preds_column)])
    # preds_df.to_csv(args.output_dir + "/" + args.run_id + "/predictions.csv", index=False)
    #
    # labels_df = pd.DataFrame(labels, columns=["label_" + str(i) for i in range(max_preds_column)])
    # labels_df.to_csv(args.output_dir + "/" + args.run_id + "/labels.csv", index=False)

    # # predict on the test set
    # preds, labels, doc_ids, all_queries, preds_without_acc = trainer.test()

    new_preds=list((np.array(preds_without_acc)> 0.4).astype(int))
    d = {'query': all_queries, 'doc_id': doc_ids,'label': new_preds, 'similiarity':preds_without_acc}

    df_doc_ids = pd.DataFrame(d)
    import pdb
    pdb.set_trace()
    df_doc_ids = df_doc_ids.groupby('query').agg(list).reset_index()
    # df_doc_ids_ones = df_doc_ids[df_doc_ids['label']==1]
    # df_doc_ids_ones = df_doc_ids_ones.groupby('query').agg(list).reset_index()
    # df_doc_ids_non_ones = df_doc_ids.groupby('query').agg(list).reset_index()
    # new_df=[]
    # for i,row in df_doc_ids_non_ones.iterrows():
    #     if all([v == 0 for v in row['label']]):
    #         highest_value=[x for _, x in sorted(zip(row['similiarity'], row['doc_id']), key=lambda pair: pair[0])]
    #         highest_value_sim=[x for x in sorted(row['similiarity'])]
    #
    #         row['label'] = [1]
    #         row[ 'doc_id'] = [highest_value[0]]
    #         row[ 'similiarity'] = [highest_value_sim[0]]
    #
    #         new_df.append(row)

    # result = pd.concat([df_doc_ids,pd.DataFrame(new_df)])

    df_doc_ids.to_csv(args.output_dir + "/" + args.run_id + "/doc_ids_test_all_results.csv", index=False, sep='\t')


    return trainer.best_ndcg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="the folder that the model is saved in.")
    parser.add_argument("--val_batch_size", default=32, type=int, required=False,
                        help="Validation and test batch size.")
    parser.add_argument("--path_to_ranked_file", default=None, type=str, required=False,
                        help="if there is a ranked file this will be the path to it. ")
    parser.add_argument("--path_to_ranked_test", default=None, type=str, required=False,
                        help="if there is a ranked test file this will be the path to it. ")
    parser.add_argument("--path_to_ranked_dev", default=None, type=str, required=False,
                        help="if there is a ranked test file this will be the path to it. ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")

    args = parser.parse_args()
    args.sacred_ex = ex

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    return ex.run()


if __name__ == "__main__":
    main()