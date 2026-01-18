<div align="center">
  <h1>CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning</h1>

  <p><em>by</em></p>

  <table>
    <tr>
      <td align="center" style="padding: 0 16px;">
        <strong>Eric Onyame</strong><sup>*</sup><br/>
        University of Virginia
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Akash Ghosh</strong><sup>*</sup><br/>
        IIT-Patna
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Subhadip Baidya</strong><br/>
        IIT-Patna
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Sriparna Saha</strong><br/>
        IIT-Patna
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Xiuying Chen</strong><br/>
        MBZUAI
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Chirag Agarwal</strong><br/>
        University of Virginia
      </td>
    </tr>
  </table>

  <p><sup>*</sup>Equal contribution. <strong>Corresponding authors:</strong> Eric Onyame, Akash Ghosh</p>
</div>

<br/>

This repository contains the code and dataset for <strong>CURE-Med</strong>, a framework for improving multilingual medical reasoning in large language models (LLMs). It includes baseline inference, curriculum-informed reinforcement learning, and supervised fine-tuning across 13 languages, including under-represented languages such as Amharic, Yoruba, and Swahili.


## Overview
While large language models (LLMs) have shown to perform well on monolingual mathematical and commonsense reasoning, they remain unreliable for multilingual medical reasoning applications, hindering their deployment in multilingual healthcare settings. We address this by first introducing CUREMED-BENCH, a high-quality multilingual medical reasoning dataset with open-ended reasoning queries with a single verifiable answer, spanning thirteen languages, including underrepresented languages such as Amharic, Yoruba, and Swahili. Building on this dataset, we propose CURE-MED, a curriculum-informed reinforcement learning framework that integrates code-switching-aware supervised fine-tuning and Group Relative Policy Optimization to jointly improve logical correctness and language stability. Across thirteen languages, our approach consistently outperforms strong baselines and scales effectively, achieving 85.21% language consistency and 54.35% logical correctness at 7B parameters, and 94.96% language consistency and 70.04% logical correctness at 32B parameters. These results support reliable and equitable multilingual medical reasoning in LLMs. 

For full details, see the paper (to be added).

## Citation
If you find this work useful, please cite: (to be added)


## Dataset
- **CUREMED-BENCH**: Included in `data.zip`. This dataset features open-ended medical reasoning queries with a single verifiable answer across 13 languages. Unzip it for use in training and evaluation scripts.

