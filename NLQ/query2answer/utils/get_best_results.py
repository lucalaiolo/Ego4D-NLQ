import argparse
import os

def find_best_result(args):
    path = f"{args['model_base_dir']}/{args['name']}/{args['model_name']}_{args['task']}_{args['fv']}_{args['max_pos_len']}_{args['predictor']}/model/"
    global_steps = [int(f.split("_")[-2]) for f in os.listdir(path=path) if "preds" in f]
    best_global_step = max(global_steps)
    best_file = next(
        f for f in os.listdir(path)
        if "preds" in f and int(f.split("_")[-2]) == best_global_step
    )
    os.rename(f"{path}{best_file}", "/content/query2answer/best_pred.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_base_dir", required=True
    )
    parser.add_argument(
        "--name", required=True
    )
    parser.add_argument(
        "--model_name", required=True
    )
    parser.add_argument(
        "--task", required=True
    )
    parser.add_argument(
        "--fv", required=True
    )
    parser.add_argument(
        "--max_pos_len", required=True
    )
    parser.add_argument(
        "--predictor", required=True
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    find_best_result(parsed_args)