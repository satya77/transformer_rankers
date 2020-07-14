from transformers import BertTokenizer, BertForSequenceClassification

from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset, preprocess_pr
from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.eval import results_analyses_tools
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
#Read dataset
train,valid, test = preprocess_pr.transform_trec2020pr_to_dfs("../data/trec2020pr")
logging.info("Data is loaded!")

#Instantiate random negative samplers (1 for training 9 negative candidates for test)
logging.info("Generating negative samples")
ns_train = negative_sampling.RandomNegativeSampler(list(train["passage"].values), 1)
ns_val = negative_sampling.RandomNegativeSampler(list(valid["passage"].values) + \
    list(train["passage"].values), 9)

logging.info("Initlizing the Tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
special_tokens_dict = {'additional_special_tokens': ['[UTTERANCE_SEP]', '[TURN_SEP]'] }
tokenizer.add_special_tokens(special_tokens_dict)

#Create the loaders for the datasets, with the respective negative samplers
dataloader = dataset.QueryDocumentDataLoader(train_df=train, val_df=valid, test_df=valid,
                                tokenizer=tokenizer, negative_sampler_train=ns_train,
                                negative_sampler_val=ns_val, task_type='classification',
                                train_batch_size=32, val_batch_size=32, max_seq_len=512,
                                sample_data=-1, cache_path="../data")
logging.info("Initlizing the DataLoader")
train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

#Use BERT to rank responses
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
# we added [UTTERANCE_SEP] and [TURN_SEP] to the vocabulary so we need to resize the token embeddings
model.resize_token_embeddings(len(dataloader.tokenizer))

#Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,train_loader=train_loader,
                                val_loader=val_loader, test_loader=test_loader,
                                num_ns_eval=9, task_type="classification", tokenizer=tokenizer,
                                validate_every_epoch=1, num_validation_instances=-1,
                                num_epochs=1, lr=0.0005, sacred_ex=None)

#Train the model
logging.info("Fitting BERT-ranker for TERC")
trainer.fit()

#Predict for test (in our example the validation set)
logging.info("Predicting")
preds, labels = trainer.test()
res = results_analyses_tools.\
    evaluate_and_aggregate(preds, labels, ['ndcg_cut_10'])

for metric, v in res.items():
    logging.info("Test {} : {:4f}".format(metric, v))