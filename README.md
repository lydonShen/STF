# Learn How to See: Collaborative Embodied Learning for Object Detection and Camera Adjusting


> Lingdong Shen, Chunlei Huo, Nuo Xu, Chaowei Han, Zichen Wang

<p align="center" >
<img src="./img/car_1.gif" width="280" height="200">
<img src="./img/car_2.gif" width="280" height="200">
<img src="./img/car_3.gif" width="280" height="200">
</p>
<br>
<p align="center" >
<img src="./img/plane_1.gif" width="300" height="200">
<img src="./img/plane_2.gif" width="300" height="200">
<img src="./img/plane_3.gif" width="300" height="200">
</p>

<figure>
<p align="center" >
<img src='./img/model.png' width=700 alt="Figure 1"/>
</p>
</figure>

## 1. Installation
This implementation is based on [Decision Transformer](https://sites.google.com/berkeley.edu/decision-transformer), [FCOS](https://github.com/tianzhi0549/FCOS), [gym](https://github.com/openai/gym) and [keras-rl](https://github.com/keras-rl/keras-rl), .

## 2. Dataset
Download address of two image datasets to create the environment: [SA](https://www.dropbox.com/s/jwusmkq90t0cq5f/SA.zip?dl=0) and [VP](https://www.dropbox.com/s/4jmdbpy0lbnyddn/VP.zip?dl=0).

Replay buffer transition data is stored in trajectories under ./data/transition/

We have obtained the offline features of all the scene images, and you can download them if you need them [Features](https://www.dropbox.com/scl/fi/bjuyq4e4tcl86qln46c52/features.7z?rlkey=giaqpqomh0by508z10bd84vr7&dl=0)

## 3. Training agent

### Object Detection Module

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x

### Camera Control Module
For the simulated airport (SA)

    python tools/train_dqn_vs.py \
        --drl-weights training_dir/ddqn_plane \
        --double

For the virtual park (VP)

    python tools/train_dqn_vsb.py \
        --drl-weights training_dir/ddqn_car \
        --double
        
### Training STF agent
Change "--game" for different datasets

    python run_dt_eod.py --seed [seed] --context_length 6 --epochs 5 --model_type 'reward_conditioned'  --game 'SA' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]

## 4. Inference
### Step
    1. Get the search file for the inference process
    2. Interpret the search file to generate a JSON file for inference
### Camera Control Module
For the simulated airport (SA)

    python tools/test_dqn_vs.py \
        --drl-weights training_dir/ddqn_plane/dqn_weights_final.h5f \
        --pickle-dir ddqn_plane_search \
        --double

For the virtual park (VP)

    python tools/test_dqn_vsb.py \
        --drl-weights training_dir/ddqn_car/dqn_weights_final.h5f \
        --pickle-dir ddqn_car_search \
        --double
### STF Control Module
Change "--game" for different datasets
    
    python run_dt_eod.py -- test -- model_path [MODEL_PATH] --seed [seed] --context_length 6 --epochs 5 --model_type 'reward_conditioned'  --game 'SA' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]

    
Obtain ground truth of each step for testing according to the corresponding camera parameters. Please modify category and the name of pickle file in get_json.py.

    python tools/get_json.py

### Object Detection Module
For the ground truth of each step generated by get_json.py, use the following command for testing.

    python tools/test_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4
## 5. Citation

```
```
