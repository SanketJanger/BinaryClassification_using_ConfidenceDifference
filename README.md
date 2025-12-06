# Binary Classification with Confidence Difference
  
=================================================================================================
  
**Group Members:** Sanket Janger & Niraj Pisal
Github Username: **SanketJanger**  &   **npisal**

**Course Project – Intro to Machine Learning**

--------------------------------------------------------------------------------------------------

## Project Overview
This repository reproduces the results from the NeurIPS 2023 paper:

**"Binary Classification with Confidence Difference"**
*Wei Wang , Lei Feng , Yuchen Jiang , Gang Niu , Min-Ling Zhang , Masashi Sugiyama*

The paper proposes a new weak-supervision setting where the model learns **without labels**, using only **confidence differences** between sample pairs:
**c(x, x′ ) = p(y ′ = +1|x′ ) − p(y = +1|x)**


This project includes:

- Full reproduction of the original ConfDiff experiments
- Additional hypotheses proposed by our team
- Supervised baselines for comparison
- Plots and analysis used in the presentation and final report

----------------------------------------------------------------------------------------------------

## How to Run Our Experiments

## **1. Reproducing the original ConfDiff results**

python main.py -mo mlp -ds mnist -uci 0 -lr 1e-3 -wd 1e-5 \
    -ep 100 -bs 256 -pretrain_bs 256 -pretrain_ep 10 \
    -me ConfDiffABS -prior 0.5 -n 15000 -run_times 5
    
    
## **2. Label Efficiency (Supervised Small Data)(Hypothesis 1)**

python experiments/supervised_small_mnist.py

This trains a supervised MLP with:
    -1000 labels
    -3000 labels
    -5000 labels
Then compares accuracy with ConfDiff-ABS.

## **3. Label Noise Robustness (Supervised Baseline)(Hypothesis 2)**

python experiments/supervised_label_noise_mnist.py

This injects:
    - 0% label noise
    - 20% label noise
    - 40% label noise

## **4. Generate both Hypothesis Plots**

python experiments/plot_hypotheses_mnist.py

Produces both H1 and H2 graphs.

---------------------------------------------------------------------------------------------------------

## Results
**ConfDiff vs Class Prior (MNIST)**
    ConfDiff-ABS performs consistently best and remains stable across priors.

**ConfDiff Across Datasets (MNIST vs Pendigits)**
    Shows robustness and generalization ability.

---------------------------------------------------------------------------------------------------------

## Hypothesis Results

H1: ConfDiff-ABS (0 labels) achieves comparable or better accuracy than supervised learning with 1000–5000 labels.

H2: ConfDiff-ABS remains stable under label noise, while supervised accuracy collapses at 40% noise.

---------------------------------------------------------------------------------------------------------

## Citation 

@inproceedings{zhang2023binary,
  title={Binary Classification with Confidence Difference},
  author={Wei Wang , Lei Feng , Yuchen Jiang , Gang Niu , Min-Ling Zhang , Masashi Sugiyama},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

=======================================XXXXXXXXXXXXXXXXXXXX================================================
