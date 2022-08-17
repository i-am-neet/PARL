
# Launch
```bash
# to PARL/examples/MADDPG/
python train.py -h
# Training
# train.py --env WHICH_ENV --num HOW MANY ROBOTS WANT TO TRAIN --model_dir SAVE MODEL TO THIS DIR
python train.py --env simple_spread_room --num 3 --model_dir "./model_0627_fineTune_no_share/"
# Evaluation
# evaluate model w/ directory ./model, --restore START EVAL --show SHOW RESULT
python train.py --num 3 --show --restore
# Training log: all logs are saved in train_log w/ name of model_dir
```