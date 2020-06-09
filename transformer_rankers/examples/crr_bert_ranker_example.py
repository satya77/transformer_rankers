from transformer_rankers.trainers.transformer_trainer import TransformerTrainer
from transformer_rankers.datasets.crr_dataset import CRRDataLoader
from transformer_rankers.datasets.preprocess_crr import read_crr_tsv_as_df
from transformer_rankers.negative_samplers.negative_sampling import RandomNegativeSampler, TfIdfNegativeSampler

from transformers import BertTokenizer, BertForSequenceClassification
from sacred.observers import FileStorageObserver
from sacred import Experiment
from IPython import embed

import torch
import pandas as pd
import argparse
import logging

ex = Experiment('BERT-ranker experiment')

@ex.main
def run_experiment(args):
    args.run_id = str(ex.current_run._id)

    train = read_crr_tsv_as_df(args.data_folder+args.task+"/train.tsv", args.sample_data)
    valid = read_crr_tsv_as_df(args.data_folder+args.task+"/valid.tsv", args.sample_data)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')    
    
    if args.negative_sampler == 'random':
        ns_train = RandomNegativeSampler(list(train["response"].values), args.num_ns_train)
        ns_val = RandomNegativeSampler(list(train["response"].values), args.num_ns_eval)
    elif args.negative_sampler == 'tf-idf':
        ns_train = TfIdfNegativeSampler(list(train["response"].values), args.num_ns_train, 
                    args.data_folder+args.task+"/indexdir_train")
        ns_val = TfIdfNegativeSampler(list(valid["response"].values) + list(train["response"].values),
                    args.num_ns_eval, 
                    args.data_folder+args.task+"/indexdir_val")

    dataloader = CRRDataLoader(args=args, train_df=train,
                                val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train,
                                negative_sampler_val=ns_val)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

    model = BertForSequenceClassification.from_pretrained('bert-base-cased')
    model.resize_token_embeddings(len(dataloader.tokenizer))

    trainer = TransformerTrainer(args, model, train_loader, val_loader, test_loader, 
                                 args.num_ns_eval)

    model_name = model.__class__.__name__
    logging.info("Fitting {} for {}{}".format(model_name, args.data_folder, args.task))
    trainer.fit()
    logging.info("Predicting")
    preds = trainer.test()

    #Saving predictions to a file
    preds_df = pd.DataFrame(preds, columns=["prediction_"+str(i) for i in range(len(preds[0]))])
    preds_df.to_csv(args.output_dir+"/"+args.run_id+"/predictions.csv", index=False)

    #Saving model to a file
    if args.save_model:
        torch.save(model.state_dict(), args.output_dir+"/"+args.run_id+"/model")

    return trainer.best_ndcg

def main():
    parser = argparse.ArgumentParser()

    # Input and output configs
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="the task to run bert ranker for")
    parser.add_argument("--data_folder", default=None, type=str, required=True,
                        help="the folder containing data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the folder to output predictions")
    parser.add_argument("--save_model", default=False, type=str, required=False,
                        help="Save trained model at the end of training.")

    #Training procedure
    parser.add_argument("--seed", default=42, type=str, required=False,
                        help="random seed")
    parser.add_argument("--num_epochs", default=100, type=int, required=False,
                        help="Number of epochs for training.")
    parser.add_argument("--max_gpu", default=-1, type=int, required=False,
                        help="max gpu used")
    parser.add_argument("--validate_epochs", default=1, type=int, required=False,
                        help="Run validation every <validate_epochs> epochs.")
    parser.add_argument("--num_validation_instances", default=-1, type=int, required=False,
                        help="Run validation for a sample of <num_validation_instances>. To run on all instances use -1.")
    parser.add_argument("--train_batch_size", default=32, type=int, required=False,
                        help="Training batch size.")
    parser.add_argument("--val_batch_size", default=32, type=int, required=False,
                        help="Validation and test batch size.")
    parser.add_argument("--num_ns_train", default=1, type=int, required=False,
                        help="Number of negatively sampled documents to use during training")
    parser.add_argument("--num_ns_eval", default=9, type=int, required=False,
                        help="Number of negatively sampled documents to use during evaluation")
    parser.add_argument("--sample_data", default=-1, type=int, required=False,
                         help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--negative_sampler", default="random", type=str, required=False,
                        help="Negative candidates sampler (['random', 'tf-idf']) ")

    #Model hyperparameters
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--lr", default=5e-6, type=float, required=False,
                        help="Learning rate.")

    args = parser.parse_args()
    args.sacred_ex = ex

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    ex.observers.append(FileStorageObserver(args.output_dir))
    ex.add_config({'args': args})
    return ex.run()

if __name__ == "__main__":
    main()