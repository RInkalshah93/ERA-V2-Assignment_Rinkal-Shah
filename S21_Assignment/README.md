# Assignment
1. On Colab (or your computer), train the GPT-2 124M model on this such that your loss is less than 0.099999
2. Share the GitHub link where we can see the training logs and sample outputs
3. Share the huggingFace app where we can see it running (add a screenshot on GitHub where huggingface output is visible)

# Introduction
The goal of this assignment is to implement GPT-2 model from scratch. Add optimization to speed up
training. Train model to reach 0.1 loss and deploy it on Huggingface.

## Train logs
    using device: cuda
    loaded 338025 tokens
    1 epoch = 20 batches
    num decayed parameter tensors: 50, with 124,354,560 parameters
    num non-decayed parameter tensors: 98, with 121,344 parameters
    using fused AdamW: True
    /content/ERA-V2-Assignment_Rinkal-Shah/S21_Assignment/train_get2_9_speedup9.py:306: UserWarning: torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
    step1 | loss: 10.163562774658203 | dt: 122.74ms | tok/sec:  133485.38 | norm: 6.50
    step2 | loss: 9.513594627380371 | dt: 122.53ms | tok/sec:  133709.53 | norm: 3.79
    step3 | loss: 9.250513076782227 | dt: 118.19ms | tok/sec:  138621.11 | norm: 4.15
    step4 | loss: 9.088885307312012 | dt: 115.91ms | tok/sec:  141349.53 | norm: 4.60
    step5 | loss: 9.00467586517334 | dt: 115.89ms | tok/sec:  141375.70 | norm: 3.54
    step6 | loss: 8.888261795043945 | dt: 116.06ms | tok/sec:  141165.15 | norm: 2.32
    step7 | loss: 8.68309211730957 | dt: 116.10ms | tok/sec:  141119.93 | norm: 2.04
    step8 | loss: 8.416431427001953 | dt: 115.94ms | tok/sec:  141317.56 | norm: 3.06
    step9 | loss: 8.161882400512695 | dt: 115.90ms | tok/sec:  141363.49 | norm: 2.46
    step10 | loss: 7.893287181854248 | dt: 115.96ms | tok/sec:  141287.05 | norm: 2.79
    step11 | loss: 7.674121379852295 | dt: 115.83ms | tok/sec:  141450.49 | norm: 1.61
    step12 | loss: 7.358696937561035 | dt: 116.05ms | tok/sec:  141186.03 | norm: 1.91
    step13 | loss: 7.287788391113281 | dt: 115.79ms | tok/sec:  141494.18 | norm: 1.66
    step14 | loss: 7.140621662139893 | dt: 115.83ms | tok/sec:  141452.24 | norm: 1.33
    step15 | loss: 6.841945171356201 | dt: 115.92ms | tok/sec:  141338.77 | norm: 1.37
    step16 | loss: 6.692824840545654 | dt: 115.94ms | tok/sec:  141315.52 | norm: 1.28
    step17 | loss: 6.532230377197266 | dt: 115.94ms | tok/sec:  141310.00 | norm: 1.09
    step18 | loss: 6.45156717300415 | dt: 115.87ms | tok/sec:  141394.61 | norm: 1.05
    step19 | loss: 6.2187089920043945 | dt: 115.96ms | tok/sec:  141285.01 | norm: 1.37
    step20 | loss: 6.387146949768066 | dt: 115.83ms | tok/sec:  141451.07 | norm: 1.36
    step21 | loss: 6.174988746643066 | dt: 116.03ms | tok/sec:  141206.92 | norm: 3.24
    step22 | loss: 6.215758323669434 | dt: 115.95ms | tok/sec:  141303.90 | norm: 1.90
    step23 | loss: 6.156937122344971 | dt: 116.11ms | tok/sec:  141112.97 | norm: 1.58
    step24 | loss: 6.06749153137207 | dt: 115.87ms | tok/sec:  141394.90 | norm: 1.04
    step25 | loss: 6.307716369628906 | dt: 115.97ms | tok/sec:  141274.85 | norm: 1.80
    step26 | loss: 6.351201057434082 | dt: 115.94ms | tok/sec:  141310.58 | norm: 1.19
    step27 | loss: 6.2062201499938965 | dt: 115.92ms | tok/sec:  141342.84 | norm: 1.21
    step28 | loss: 6.127079963684082 | dt: 115.82ms | tok/sec:  141465.63 | norm: 1.36
    step29 | loss: 6.002640724182129 | dt: 116.00ms | tok/sec:  141245.23 | norm: 0.96
    step30 | loss: 5.9997687339782715 | dt: 115.91ms | tok/sec:  141346.62 | norm: 1.53
    ....
    step4971 | loss: 0.0016334668034687638 | dt: 116.26ms | tok/sec:  140930.36 | norm: 0.01
    step4972 | loss: 0.0008988007321022451 | dt: 116.48ms | tok/sec:  140654.60 | norm: 0.01
    step4973 | loss: 0.001018062001094222 | dt: 116.77ms | tok/sec:  140306.54 | norm: 0.01
    step4974 | loss: 0.0013380665332078934 | dt: 116.74ms | tok/sec:  140347.52 | norm: 0.01
    step4975 | loss: 0.0008820893126539886 | dt: 116.22ms | tok/sec:  140969.68 | norm: 0.01
    step4976 | loss: 0.0007170508615672588 | dt: 116.14ms | tok/sec:  141075.31 | norm: 0.01
    step4977 | loss: 0.00148543540854007 | dt: 116.66ms | tok/sec:  140439.01 | norm: 0.01
    step4978 | loss: 0.001079896348528564 | dt: 116.65ms | tok/sec:  140452.22 | norm: 0.01
    step4979 | loss: 0.0012477301061153412 | dt: 117.00ms | tok/sec:  140038.35 | norm: 0.01
    step4980 | loss: 0.0010526750702410936 | dt: 116.48ms | tok/sec:  140654.31 | norm: 0.01
    step4981 | loss: 0.0014646396739408374 | dt: 116.20ms | tok/sec:  141003.81 | norm: 0.01
    step4982 | loss: 0.0014233369147405028 | dt: 116.47ms | tok/sec:  140666.69 | norm: 0.01
    step4983 | loss: 0.0014269333332777023 | dt: 116.99ms | tok/sec:  140042.91 | norm: 0.01
    step4984 | loss: 0.0018635840388014913 | dt: 116.37ms | tok/sec:  140793.21 | norm: 0.01
    step4985 | loss: 0.0009708877769298851 | dt: 116.38ms | tok/sec:  140782.54 | norm: 0.01
    step4986 | loss: 0.0009857664117589593 | dt: 116.50ms | tok/sec:  140639.05 | norm: 0.01
    step4987 | loss: 0.0020635968539863825 | dt: 116.88ms | tok/sec:  140177.17 | norm: 0.01
    step4988 | loss: 0.0013513314770534635 | dt: 116.95ms | tok/sec:  140089.16 | norm: 0.01
    step4989 | loss: 0.0014694007113575935 | dt: 116.32ms | tok/sec:  140854.39 | norm: 0.01
    step4990 | loss: 0.0012462016893550754 | dt: 116.02ms | tok/sec:  141212.14 | norm: 0.03
    step4991 | loss: 0.0016851801192387938 | dt: 116.60ms | tok/sec:  140512.52 | norm: 0.01
    step4992 | loss: 0.0008807081612758338 | dt: 116.63ms | tok/sec:  140478.92 | norm: 0.01
    step4993 | loss: 0.0010219431715086102 | dt: 116.64ms | tok/sec:  140463.70 | norm: 0.01
    step4994 | loss: 0.0013367274077609181 | dt: 116.13ms | tok/sec:  141088.34 | norm: 0.01
    step4995 | loss: 0.0008769867708906531 | dt: 116.22ms | tok/sec:  140975.17 | norm: 0.01
    step4996 | loss: 0.0006504447082988918 | dt: 116.48ms | tok/sec:  140657.77 | norm: 0.01
    step4997 | loss: 0.001547040417790413 | dt: 116.61ms | tok/sec:  140504.48 | norm: 0.01
    step4998 | loss: 0.001045654178597033 | dt: 116.67ms | tok/sec:  140427.82 | norm: 0.01
    step4999 | loss: 0.0013870284892618656 | dt: 116.24ms | tok/sec:  140952.62 | norm: 0.01
    tensor(0.0014, device='cuda:0', grad_fn=<NllLossBackward0>)


## Metrics
Final loss: 0.0013

## Gradio App
![Gradio-app](./images/gradio_app.png)  
Gradio App can be found [here](https://huggingface.co/spaces/AkashDataScience/GPT-2)

## Acknowledgments
This model is trained using repo listed below
* [GPT2](https://github.com/RInkalshah93/ERA-V2-Assignment_Rinkal-Shah/tree/7f99c19c4dab9c6b5327b700d582e0b08e85ea4d/S21_Assignment)