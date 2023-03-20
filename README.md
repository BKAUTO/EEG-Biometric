# EEG-Biometric
This is the repo for EEG-based Authentication research of [SPTAGE Lab](https://sptage.compute.dtu.dk/).

The project includes an end-to-end system for EEG Authentication. Different ML/DL models are included for comparison. 

Data processing logic for two datasets ([PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) and [BCI IV 2a](https://www.bbci.de/competition/iv/)) are implemented in `/util`. It is possible to input other EEG datasets according to their formats.

Here are the essential dependencies:
```
Python             3.9.6
numba              0.54.0
numpy              1.20.3
torch              1.9.0
```

Please cite the correponding paper:
> @inproceedings{wu2022towards,
>   title={Towards Enhanced EEG-based Authentication with Motor Imagery Brain-Computer Interface},
>   author={Wu, Bingkun and Meng, Weizhi and Chiu, Wei-Yang},
>   booktitle={Proceedings of the 38th Annual Computer Security Applications Conference},
>   pages={799--812},
>   year={2022}
> } 