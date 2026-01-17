# CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning

This repository contains the code and dataset for the CURE-Med framework, focusing on improving multilingual medical reasoning in large language models (LLMs). It includes implementations for baseline inference, curriculum-informed reinforcement learning, and supervised fine-tuning, supporting 13 languages including underrepresented ones like Amharic, Yoruba, and Swahili.

## Authors
<center> Eric Onyame\*, Akash Ghosh\*, Subhadip Baidya, Sriparna Saha, Xiuying Chen, Chirag Agarwal </center> 
<!-- - Eric Onyame\* (University of Virginia)
- Akash Ghosh\* (IIT-Patna)
- Subhadip Baidya (IIT-Patna)
- Sriparna Saha (IIT-Patna)
- Xiuying Chen (MBZUAI)
- Chirag Agarwal (University of Virginia) -->

\*Equal Contribution. Correspondence Authors: Eric Onyame and Akash Ghosh

## Overview
While large language models (LLMs) have shown to perform well on monolingual mathematical and commonsense reasoning, they remain unreliable for multilingual medical reasoning applications, hindering their deployment in multilingual healthcare settings. We address this by first introducing CUREMED-BENCH, a high-quality multilingual medical reasoning dataset with open-ended reasoning queries with a single verifiable answer, spanning thirteen languages, including underrepresented languages such as Amharic, Yoruba, and Swahili. Building on this dataset, we propose CURE-MED, a curriculum-informed reinforcement learning framework that integrates code-switching-aware supervised fine-tuning and Group Relative Policy Optimization to jointly improve logical correctness and language stability. Across thirteen languages, our approach consistently outperforms strong baselines and scales effectively, achieving 85.21% language consistency and 54.35% logical correctness at 7B parameters, and 94.96% language consistency and 70.04% logical correctness at 32B parameters. These results support reliable and equitable multilingual medical reasoning in LLMs. 

For full details, see the paper (to be added).

## Citation
If you find this work useful, please cite: (to be added)


## Dataset
- **CUREMED-BENCH**: Included in `data.zip`. This dataset features open-ended medical reasoning queries with a single verifiable answer across 13 languages. Unzip it for use in training and evaluation scripts.

