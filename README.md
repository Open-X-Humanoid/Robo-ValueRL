# Robo-Valuerl

[Project Page](https://gewu-lab.github.io/robo-valuerl/) | [Paper]() | [Model](https://huggingface.co/X-Humanoid/Robo-ValueRL) | [Dataset](https://huggingface.co/datasets/X-Humanoid/Robo-ValueRL)

## Environment Setup

We have build a complete environment for Robo-Valuerl. You can install it by running the following command:

```bash
bash install_simple.sh
```

## Usage

You can use the following command to train the Robo-Valuerl value estimator:

```bash
cd robo_valuerl
bash scripts/train_robo_valuerl_value_estimator.sh
```

Further, the value estimator is used to estimate the remain time of data, and convert it to the action quality:

```bash
cd robo_valuerl
python annotate_value_function_sparse_part.py all_value_function_training --exp_name test --pytorch-weight-path value_estimator_model_path --annotation_dir_root offline_data_dir --annotate_total 2 --annotate_count 0 --history_length 5
```

After the annotation, you can use the following command to generate the quality:
```bash
cd robo_valuerl
python generate_quality.py --data_dir offline_data_dir
```

Finally, you can use the following command to train the Robo-Valuerl offline pretraining:

```bash
cd robo_valuerl
bash scripts/train_robo_valuerl_offline.sh
```

After we collect the online rollout dataset, we could use the following command to train the Robo-Valuerl online adaptation:

```bash
cd robo_valuerl
bash scripts/train_robo_valuerl_online.sh
```

