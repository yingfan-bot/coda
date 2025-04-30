# Requirements
See requirements.txt.

# Instructions to run

For original antmaze-v2: (replacing the env name with any of the six antmaze-v2 envs)
```
cd ./cgo
python main.py --env-name "antmaze-umaze-diverse-v2"
```

For Four Rooms: (replacing the env name with any of the six antmaze-v2 envs)
```
cd ./cgo
python main_four_rooms.py --env-name "antmaze-medium-play-v2"
```

For Random Cells: (replacing the env name with any of the six antmaze-v2 envs)
```
cd ./cgo
python main_cell.py --env-name "antmaze-medium-play-v2"
```
If you find the code useful please cite:

```
@inproceedings{
fan2024how,
title={How to Solve Contextual Goal-Oriented Problems with Offline Datasets?},
author={Ying Fan and Jingling Li and Adith Swaminathan and Aditya Modi and Ching-An Cheng},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=Ku31aRq3sW}
}
```
