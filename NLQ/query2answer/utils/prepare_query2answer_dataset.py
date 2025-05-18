import argparse
import json
import os

def compute_tIoU(predicted_start_time, predicted_end_time, real_start_time, real_end_time):
    '''
    Compute the tIoU between two time intervals.

    :param predicted_start_time: start time of the predicted interval (scalar)
    :param predicted_end_time: end time of the predicted interval (scalar)
    :param real_start_time: start time of the real interval (scalar)
    :param real_end_time: end time of the real interval (scalar)

    :return: tIoU
    '''
    intersection_lb = max(predicted_start_time, real_start_time)
    intersection_ub = min(predicted_end_time, real_end_time)
    intersection = max(0, intersection_ub - intersection_lb)

    union_lb = min(predicted_start_time, real_start_time)
    union_ub = max(predicted_end_time, real_end_time)
    union = max(0, union_ub - union_lb)

    assert union > 0, "Null union"

    return intersection / union

def reformat_gt(gt_path):
    '''
    Reformat the data to match the format required by the query2answer function.
    The path to the input json file is expected to follow the structure of the validation data file.

    :param gt_path: path to the input json file

    :return: path to the reformatted data
    '''
    with open(gt_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    val_data = val_data["videos"]
    reformat_data = []
    for vid in val_data:
        for clip in vid["clips"]:
            clip_uid = clip["clip_uid"]
            query_idx = 0
            for annotation in clip["annotations"]:
                for query in annotation["language_queries"]:
                    s_time = query["clip_start_sec"]
                    e_time = query["clip_end_sec"]
                    q = query["query"]
                    reformat_data.append({
                        "clip_uid": clip_uid, 
                        "query_idx": query_idx, 
                        "s_time": s_time, 
                        "e_time": e_time,
                        "query": q,
                    })
                    query_idx += 1
    new_path = gt_path.replace(".json", "_reformat.json")
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(reformat_data, f)
    return new_path

def get_top_predictions(predictions_path, gt_path, n):
    '''
    Select the top-n queries whose predicted video segments best match the ground-truth segments,
    ranked by tIoU.

    :param predictions_path: path to the predictions json file
    :param gt_path: path to the ground truth json file
    :param n: number of queries to select

    :return: list of top-n queries
    '''
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    queries_list = []
    for result,  gt in zip(predictions["results"], ground_truth):
        assert result["query_idx"] == gt["query_idx"] and result["clip_uid"] == gt["clip_uid"],\
            "Mismatch between predictions and ground truth"
        clip_uid = result["clip_uid"]
        query_idx = result["query_idx"]
        predicted_start_time, predicted_end_time = result["predicted_times"][0][0], result["predicted_times"][0][1]

        real_start_time, real_end_time = gt["s_time"], gt["e_time"]
        tIoU = compute_tIoU(predicted_start_time, predicted_end_time, real_start_time, real_end_time)
        query_dict = {
            "clip_uid": clip_uid,
            "query_idx": query_idx,
            "query": gt["query"],
            "tIoU": tIoU,
            "predicted_start_time": predicted_start_time,
            "predicted_end_time": predicted_end_time,
            "real_start_time": real_start_time,
            "real_end_time": real_end_time,
        }
        queries_list.append(query_dict)
    queries_list.sort(key=lambda x: x['tIoU'], reverse=True)
    if len(queries_list) <= n:
        return queries_list
    else:
        return queries_list[:n]

def retrieve_queries_and_clips(args):
    predictions_path = args["predictions_path"]
    gt_path = args["ground_truth_path"]
    # Reformat the file containing the ground truth
    gt_path = reformat_gt(gt_path)
    # Get the top-n predictions
    top_predictions = get_top_predictions(predictions_path, gt_path, n=args["n"])
    # Retrieve the clips to download
    clips_to_download = list(set([pred["clip_uid"] for pred in top_predictions]))
    # Upload the json file with the clip_uids to download
    clips_to_download = {"clip_uid": clips_to_download}
    with open(os.path.join(args["output_save_path"], "clips.json"), "w", encoding="utf-8") as f:
        json.dump(clips_to_download, f)
    # Upload the json file with the queries to answer
    to_answer = {"data": top_predictions}
    with open(os.path.join(args["output_save_path"], "top_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(to_answer, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_path", required=True, help="Path to the json file containing the predictions"
    )
    parser.add_argument(
        "--ground_truth_path", required=True, help="Path to the json file containing the ground truth"
    )
    parser.add_argument(
        "--n", required=True, help="Number of queries to answer", type=int
    )
    parser.add_argument(
        "--output_save_path", required=True, help="Path to the output directory containing the clips and queries to analyze"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    retrieve_queries_and_clips(parsed_args)