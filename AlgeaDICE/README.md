# AlgaeDICE

PyTorch Code Implementation for AlgaeDICE as described in  the paper: 

* `AlgaeDICE: Policy Gradient from Arbitrary
Experience' by Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and
Dale Schuurmans.

* Paper available on arXiv [here](https://arxiv.org/abs/1912.02074).

* Original code implementation in Tensorflow is [here](https://github.com/google-research/google-research/tree/master/algae_dice)

You can site the code base:
```
@misc{pytorchrl,
  author = {Arnob, SY},
  title = {PyTorch Implementations of DICE Algorithms},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SaminYeasar/PyTorch-implementation-DICE-algorithms}},
}
```

## Basic Commands

Run AlgaeDICE on HalfCheetah:

```
python -m algae_dice.train_eval --logtostderr --save_dir=$HOME/algae/ \
    --env_name=HalfCheetah-v2 --seed=42
```
## Important tricks

* Doubel-Q learning and Mixed critic update is important for training algeaDICE
* Unlike original implementation, there's no separate buffer to store initial states, here we can consider each state as initial state to the agent. Similar assumption is made in [here] (https://arxiv.org/abs/1912.05032)

## Performance comparison with the original implementation

![](https://imgur.com/WtrSzs3.png)
![](https://imgur.com/xy010PV.png)
![](https://imgur.com/X5XCRHQ.png)
![](https://imgur.com/ZbyUnPw.png)
