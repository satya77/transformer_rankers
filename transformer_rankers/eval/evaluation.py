from IPython import embed
import pytrec_eval
import  numpy as np
from sklearn.metrics import precision_recall_fscore_support

METRICS = {'map',
           'recip_rank',
           'ndcg_cut',
           'recall'}

RECALL_AT_W_CAND = {
                    'R_10@1',
                    'R_10@2', 
                    'R_10@5',
                    'R_2@1'
                    }

def recall_at_with_k_candidates(preds, labels, k, at):
    """
    Calculates recall with k candidates. labels list must be sorted by relevance.

    Args:
        preds: float list containing the predictions.
        labels: float list containing the relevance labels.
        k: number of candidates to consider.
        at: threshold to cut the list.
        
    Returns: float containing Recall_k@at
    """
    num_rel = labels.count(1)
    #'removing' candidates (relevant has to be in first positions in labels)
    preds = preds[:k]
    labels = labels[:k]

    sorted_labels = [x for _,x in sorted(zip(preds, labels), reverse=True)]
    hits = sorted_labels[:at].count(1)

    return hits/num_rel

def relevancy_accuracy(labels,preds,threshold=0.3):
    '''
    calculates the accuracy of the relavant results, from all the top k how many of them are really relevant
    :param labels: true labels
    :param preds: predictions
    :return: accuracy
    '''

    num_rel =[label.count(1) for label in labels]
    pred_rel=[]
    for p in preds:
        pred_rel.append(np.sum(np.array(p)>threshold))
    return np.sum(np.array(num_rel)==np.array(pred_rel))/len(pred_rel)

def relevancy_accuracy_upto(labels,preds,threshold=0.3,upto=1):
    '''
    calculates the accuracy of the relavant results, from all the top k how many of them are really relevant allowing upto mistakes
    :param labels: true labels
    :param preds: predictions
    :return: accuracy
    '''

    num_rel =[label.count(1) for label in labels]
    correct=0
    for p,label_sum in zip(preds,num_rel):
        pred_rel=np.sum(np.array(p)>threshold)
        if pred_rel- label_sum <=upto:
            correct+=1
    return correct/len(num_rel)

def relevancy_precision_recall(labels,preds,threshold=0.3):
    '''
    calculates the precision and recall of the relavant results
    :param labels:
    :param preds:
    :return: accuracy
    '''

    true_positive = 0
    total_positive = 0
    total_predicted_positive = 0
    labels_flat=[]
    pred_flat=[]
    for pred,label in zip(preds,labels):
        for pr,rr in zip(pred,label):
            pr = int(pr > threshold)
            pred_flat.append(pr)
            labels_flat.append(rr)
            if pr == 1 and rr== 1:
                true_positive += 1
            if pr == 1:
                total_predicted_positive += 1
            if rr == 1:
                total_positive += 1

    precsion = true_positive/float(total_predicted_positive)
    recall = true_positive/float(total_positive)
    f_score = 2 * ((precsion * recall)/(precsion+recall))
    return precsion,recall,f_score



def evaluate_models(results):
    """
    Calculate METRICS for each model in the results dict
    
    Args:
        results: dict containing one key for each model and inside them pred and label keys. 
        For example:    
             results = {
              'model_1': {
                 'preds': [[1,2],[1,2]],
                 'labels': [[1,2],[1,2]]
               }
            }.
    Returns: dict with the METRIC results per model and query.
    """    

    for model in results.keys():
        preds = results[model]['preds']
        labels = results[model]['labels']
        run = {}
        qrel = {}
        for i, p in enumerate(preds):
            run['q{}'.format(i+1)] = {}
            qrel['q{}'.format(i+1)] = {}
            for j, _ in enumerate(range(len(p))):
                run['q{}'.format(i+1)]['d{}'.format(j+1)] = float(preds[i][j])
                qrel['q{}'.format(i + 1)]['d{}'.format(j + 1)] = int(labels[i][j])        
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, METRICS)
        results[model]['eval'] = evaluator.evaluate(run)

        results[model]['accuracy_0.3'] = relevancy_accuracy(labels,preds,0.3)
        results[model]['accuracy_0.3_upto_1'] = relevancy_accuracy_upto(labels,preds,0.3, 1)
        results[model]['accuracy_0.4'] = relevancy_accuracy(labels,preds,0.4)
        results[model]['accuracy_0.4_upto_1'] = relevancy_accuracy_upto(labels,preds,0.4,1)

        results[model]['accuracy_0.5'] = relevancy_accuracy(labels,preds,0.5)
        results[model]['accuracy_0.5_upto_1'] = relevancy_accuracy_upto(labels,preds,0.5,1)

        results[model]['precision_0.3'] , results[model]['recall_0.3'], results[model]['f_score_0.3'] = relevancy_precision_recall(labels,preds,0.3)
        results[model]['precision_0.4'] , results[model]['recall_0.4'], results[model]['f_score_0.4'] = relevancy_precision_recall(labels,preds,0.4)
        results[model]['precision_0.5'] , results[model]['recall_0.5'], results[model]['f_score_0.5'] = relevancy_precision_recall(labels,preds,0.5)

        for query in qrel.keys(): 
            preds = []
            labels = []
            for doc in run[query].keys():
                preds.append(run[query][doc])
                labels.append(qrel[query][doc])
            
            for recall_metric in RECALL_AT_W_CAND:
                cand = int(recall_metric.split("@")[0].split("R_")[1])
                at = int(recall_metric.split("@")[-1])
                results[model]['eval'][query][recall_metric] = recall_at_with_k_candidates(preds, labels, cand, at)
    return results