**Paper Title: Binary Classification Using Confidence Difference**

Group Members: 1. **Sanket Janger** | B01097463 |      2. **Niraj Pisal** | B01099881

*********Contributions*********

Sanket Janger: 
1. Implemented and analyzed the ConfDiff-Unbiased, ConfDiff-ReLU, and ConfDiff-ABS estimators on MNIST and Pendigits.
2. Generated accuracy results across different class priors (π = 0.2, 0.5, 0.8) and validated consistency with the NeurIPS paper.
3. Implemented all supervised baseline experiments used for hypothesis testing, including:
    - small-data experiment (1k, 3k, 5k labels)
    - label-noise robustness experiment (0%, 20%, 40% noise)
4. Developed the core structure of the technical report and presentation slides, including background, methodology, and main result interpretation.
5. Ensured proper documentation and GitHub organization of the experiments folder.


Niraj Pisal: 

1. Wrote and organized custom Python scripts:
    - supervised_small_mnist.py
    - supervised_label_noise_mnist.py
    - plot_hypotheses_mnist.py
2. Generated the new hypothesis result graphs:
    - Supervised accuracy vs number of labels
    - Supervised accuracy vs label noise
    - Contributed significantly to the Discussion and Hypothesis Result sections in the report.
3. Helped design the hypothesis structure and ensured experimental alignment with the paper.
4. Led the reproduction of the original ConfDiff experiments using the authors' official codebase.

               

