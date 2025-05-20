import argparse
import json
import torch

def add_gt_answers(data_path, gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        answers = [a.strip() for a in content.split(";") if a]
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["data"]
    assert len(data) == len(answers)
    for idx, record in enumerate(data):
        gt_answer = answers[idx]
        record["answer"] = gt_answer
        data[idx] = record
    data = {"data": data}
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def scoring(args):
    data_path = args["data_path"]
    scorer = args["scorer"]
    output_save_path = args["output_save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add the ground truth answers to our data
    add_gt_answers(data_path, args["gt_answers_path"])

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    data = data["data"]

    if scorer == "bert-score":
        from bert_score import BERTScorer
        scorer_fun = BERTScorer(model_type='bert-base-uncased', device=device)
    else:
        raise NotImplementedError(f"Scorer {scorer} has not been implemented yet.")
    
    out_data = []
    precision_ = []
    recall_ = []
    f1_score_ = []
    with torch.no_grad():
        for record in data:
            gt_answer = record["answer"]
            gen_answer = record["model_answer"]
            precision, recall, f1_score = scorer_fun.score([gen_answer], [gt_answer])
            record["precision"] = precision.item()
            record["recall"] = recall.item()
            record["f1_score"] = f1_score.item()
            out_data.append(record)
            precision_.append(precision.item())
            recall_.append(recall.item())
            f1_score_.append(f1_score.item())
    
    mean_precision = sum(precision_) / len(precision_)
    mean_recall = sum(recall_) / len(recall_)
    mean_f1_score = sum(f1_score_) / len(f1_score_)

    out_data = {
        "data": out_data,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1_score": mean_f1_score
    }

    with open(output_save_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path", required=True, help="path to the json file containing the generated answers to the queries"
    )
    parser.add_argument(
        "--gt_answers_path", required=True, help="path to the .txt file containing the ground truth answers to the queries"
    )
    parser.add_argument(
        "--scorer", default="bert-score", help="scorer function to compute similarity between true and generated answers"
    )
    parser.add_argument(
        "--output_save_path", required=True, help="path to the output json file"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    scoring(parsed_args)