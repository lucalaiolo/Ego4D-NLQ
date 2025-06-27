# Natural Language Queries for Egocentric Vision 
## Machine Learning for Mathematical Engineering 2024/2025 
### Politecnico di Torino

Code for the _Natural Language Queries for Egocentric Vision_ project by Bocco, Cheraz, Di Felice and Laiolo.

## Setup
It is assumed that you are working on Colab.

To access the Ego4d dataset you need to sign the license as described [here](https://ego4d-data.org/docs/start-here/#license-agreement). 

After obtaining the AWS access credentials, you can download the dataset using the following commands.
```
import os
os.environ['AWS_ACCESS_KEY_ID'] = "< INSERT YOUR AWS ACCESS KEY ID HERE >"
os.environ['AWS_SECRET_ACCESS_KEY'] = "< INSERT YOUR AWS SECRET ACCESS KEY HERE >"
```
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip >/dev/null
sudo ./aws/install >/dev/null 2>&1
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" && aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
rm "awscliv2.zip"

pip install ego4d

ego4d --output_directory="/content/ego4d_data/" --version v1 --datasets annotations omnivore_video_swinl_fp16 --benchmarks nlq -y
```
This downloads the annotations for the NLQ benchmark together with Omnivore video features.

To download EgoVLP features:
```
pip install -U gdown

gdown --id 1TXBlLDqDuL_XPCuXlgiikEfVXO8Ly6eM -O /content/EgoVLP_train_val_features.gz
gunzip /content/EgoVLP_train_val_features.gz
mkdir -p /content/EgoVLP_train_val
tar -xf /content/EgoVLP_train_val_features -C /content/EgoVLP_train_val

gdown --id 1-CGZg9t-kpW5bmg9M62VHk5eYTllsPKV -O /content/EgoVLP_test_features.gz
gunzip /content/EgoVLP_test_features.gz
mkdir -p /content/EgoVLP_test
tar -xf /content/EgoVLP_test_features -C /content/EgoVLP_test

mkdir -p /content/ego4d_data/v1/EgoVLP
mv /content/EgoVLP_train_val/apdcephfs/private_qinghonglin/video_dataset/ego4d/benchmark_splits/nlq/nips/egovlp_egonce/* /content/ego4d_data/v1/EgoVLP
mv /content/EgoVLP_test/egovlp_egonce_test/* /content/ego4d_data/v1/EgoVLP
rm -r /content/EgoVLP_train_val/
rm -r /content/EgoVLP_test/
rm -r /content/EgoVLP_train_val_features
rm -r /content/EgoVLP_test_features
```

To download the checkpoints for the EgoVLP text encoder run the following python cells:
```
!gdown --id 1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7 -O /content/EgoVLP_checkpoints.pth
```
```
%%bash
git clone https://github.com/showlab/EgoVLP
cd EgoVLP
git pull
```
```
!mkdir -p /content/text_encoder_weights
text_encoder_weights_path = "/content/text_encoder_weights"
import torch
import sys
sys.path.insert(0, '/content/EgoVLP')
checkpoint = torch.load("/content/EgoVLP_checkpoints.pth", map_location=torch.device('cpu'), weights_only=False)
del sys.path[0]
checkpoint = {k.replace("module.text_model.", ""): v for k, v in checkpoint['state_dict'].items() if "text_model" in k}
torch.save(checkpoint, os.path.join(text_encoder_weights_path, "weights.pth"))
!rm -r /content/EgoVLP
```

Finally, clone our repository
```
git clone https://github.com/lucalaiolo/Ego4D-NLQ
cd Ego4D-NLQ
git pull
```

Please note that:
- NLQ annotations have a known issue where ~14% of annotations have a near-0 query window and will result in under reported performance for the challenge (which will be corrected with a future dataset update): [NLQ Forum Post](https://discuss.ego4d-data.org/t/nlq-annotation-zero-temporal-windows/36)

## Training

This repository currently provides the VSLNet model only. With a few on-the-fly code tweaks, you can also reproduce VSLBase (and make the video-text encoder not shared, too).

In the following cells, we give instructions on how to train VSLNet on EgoVLP features.
```
with open("vars.sh", "w") as out_f:
  out_f.write("""
export NAME=egovlp
export TASK_NAME=nlq_official_v1_$NAME
export BASE_DIR=data/dataset/nlq_official_v1_$NAME
export FEATURE_BASE_DIR=data/features/nlq_official_v1_$NAME/
export FEATURE_DIR=$FEATURE_BASE_DIR/video_features
export MODEL_BASE_DIR=/content/nlq_official_v1/checkpoints/
export PREDICTOR=EgoVLP
export FV=official
export MAX_POS_LEN=128
export MODEL_NAME=vslnet

cd Ego4D-NLQ/NLQ/VSLNet
"""
  )
```

```
%%bash

source vars.sh

echo $FEATURE_BASE_DIR
mkdir -p $FEATURE_BASE_DIR
ln -s /content/ego4d_data/v1/EgoVLP $FEATURE_DIR
```

```
%%bash
%%capture

source vars.sh
pip install nltk submitit torch torchaudio torchvision tqdm transformers tensorboard Pillow terminaltables
```

> ⚠️ When using **EgoVLP** clip features, add `--use_clip_features_only` to the python command of the cell below. We must specify this because EgoVLP features are __clip-level features__, while Omnivore features are __video-level features__.


```
%%bash

source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split /content/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split /content/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split /content/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR \
    --use_clip_features_only
```

The original repository we forked supported a Tensorboard interface to visualize the training process. We decided to keep it.
```
%load_ext tensorboard

!mkdir -p /content/Ego4D-NLQ/NLQ/VSLNet/runs/
%tensorboard --logdir /content/Ego4D-NLQ/NLQ/VSLNet/runs/egovlp_bs32_dim128_epoch10_ilr0.0025/
```

Finally, to train the model we have to run the following:

```
%%bash

source vars.sh

# machine parameters
export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"

# hyper parameters
export BATCH_SIZE=32
export DIM=128

export NUM_EPOCH=15
export INIT_LR=0.0025

export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"

python main.py \
    --task $TASK_NAME \
    --predictor $PREDICTOR \
    --dim $DIM \
    --mode train \
    --video_feature_dim 256 \
    --max_pos_len $MAX_POS_LEN \
    --init_lr $INIT_LR \
    --epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --fv $FV \
    --num_workers $NUM_WORKERS \
    --data_loader_workers $DATALOADER_WORKERS \
    --model_dir $MODEL_BASE_DIR/$NAME \
    --eval_gt_json $VAL_JSON_PATH \
    --model_name $MODEL_NAME \
    --log_to_tensorboard $TB_LOG_NAME \
    --tb_log_freq 5 \
    --remove_empty_queries_from train \
    --EgoVLP_text_encoder_weights_path /content/text_encoder_weights/weights.pth
```

> ⚠️ When using **EgoVLP** features, `video_feature_dim` must be set to 256. If using **Omnivore** features, `video_feature_dim` must be set to 1536.

## Extension - From video interval to a textual answer
The NLQ task outputs the time interval during which the input query is answered, but it's essential to note that to have the actual answer a person should then watch that interval.

This extension tackles these challenges by using one of the models trained as the initial stage of a video question-answering pipeline. First, it locates the most relevant segments in lengthy videos, then feeds those segments into a vision-language model (VLM) to generate textual answers. Specifically, we used [Video-LLaVA](https://huggingface.co/docs/transformers/model_doc/video_llava).

Install dependencies:
```
pip install --upgrade -q accelerate bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install -q av
pip install ffmpeg-python
pip install bert-score
```
First of all, we must retrieve the path to the predictions file of the best model we have trained. Such file will appear inside `/content/query2answer/best_pred.json`. In particular, this file will contain the predictions on the **validation data**.
```
%%bash
source vars.sh
cd /content/Ego4D-NLQ/NLQ/query2answer

mkdir -p /content/query2answer/

python utils/get_best_results.py \
    --model_base_dir $MODEL_BASE_DIR \
    --name $NAME \
    --model_name $MODEL_NAME \
    --task $TASK_NAME \
    --fv $FV \
    --max_pos_len $MAX_POS_LEN \
    --predictor $PREDICTOR
```
After this, we reformat the validation dataset and retrieve the desired number of queries to answer and the corresponding clips to download.

```
%%bash
source vars.sh
cd /content/Ego4D-NLQ/NLQ/query2answer

export NUM_TO_ANSWER=50

export VAL_JSON_PATH="/content/query2answer/val.json"
export OUTPUT_SAVE_PATH="/content/query2answer/"

python utils/prepare_query2answer_dataset.py \
    --predictions_path /content/query2answer/best_pred.json \
    --ground_truth_path $VAL_JSON_PATH \
    --n $NUM_TO_ANSWER \
    --output_save_path $OUTPUT_SAVE_PATH
```
The variable `VAL_JSON_PATH` must be changed accordingly.

Next, we download the selected clips

```
import json
clips_path = "/content/clips (1).json"
#clips_path = "/content/query2answer/clips.json"
with open(clips_path, "r") as f:
    clips_to_download = json.load(f)
clips_to_download = clips_to_download["clip_uid"]
clips_to_download = " ".join(clips_to_download)
!mkdir -p /content/query2answer/clips/
!ego4d --output_directory="/content/query2answer/clips/" --version v1 --dataset clips --video_uids $clips_to_download
```

We can finally use the Video-LLaVA model to obtain the answers to our queries. The answers will be stored in  `/content/query2answer/answers.json`
```
%%bash

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

source vars.sh
cd /content/Ego4D-NLQ/NLQ/query2answer

python main.py \
    --data_path /content/query2answer/top_predictions.json \
    --clips_path /content/query2answer/clips/v1/clips/ \
    --output_path /content/query2answer/answers.json
```

To assess the quality of the generated answers, we first have to manually annotate the actual answers for each query. To do so, it is necessary to watch each video clip.

Before running the next cell, upload the file containing the ground truth answers to `/content/query2answer/gt_answers.txt`. This file should be organized in the following way:


```
ANSWER_0;ANSWER_1;_ANSWER_2;...
```

The file containing the final answers toghether with the computed scores can be found at `/content/query2answer/final_answers.json`.
To compare the ground truth and generated answers we use [BERTScore](https://github.com/Tiiiger/bert_score).

```
%%bash

source vars.sh
cd /content/Ego4D-NLQ/NLQ/query2answer

python utils/compute_score.py \
    --data_path /content/query2answer/answers.json \
    --gt_answers_path /content/query2answer/gt_answers.txt \
    --scorer bert-score \
    --output_save_path /content/query2answer/final_answers.json
```
