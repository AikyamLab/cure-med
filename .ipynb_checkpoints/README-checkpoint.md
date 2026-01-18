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

This repository hosts the codebase and dataset for <strong>CURE-Med</strong>, a framework for improving multilingual medical reasoning in large language models (LLMs). Below, we provide an overview of the project along with key training and implementation details.



## Overview
Large language models (LLMs) perform strongly on monolingual math and commonsense reasoning, but they remain unreliable for multilingual medical reasoning—limiting safe use in real-world, multilingual healthcare settings. To address this, we introduce <strong>CUREMED-BENCH</strong>, a high-quality multilingual medical reasoning benchmark of open-ended questions with a single verifiable answer, spanning 13 languages, including under-represented languages such as Amharic, Yoruba, and Swahili. Building on this benchmark, we propose <strong>CURE-MED</strong>, a curriculum-informed reinforcement learning framework that combines code-switching-aware supervised fine-tuning with Group Relative Policy Optimization to improve both logical correctness and language stability. Across 13 languages, CURE-MED consistently outperforms strong baselines and scales effectively, reaching 85.21% language consistency and 54.35% logical correctness at 7B parameters, and 94.96% language consistency and 70.04% logical correctness at 32B parameters. Overall, our results move toward more reliable and equitable multilingual medical reasoning with LLMs.


## Key Figure

<p align="center">
  <img src="figures/cure_med.png" alt="CURE-MED pipeline overview" width="900">
  <br/>
  <em><strong>Figure 1.</strong> CURE-MED pipeline: (A) clinically validated multilingual data curation (e.g., MedlinePlus), (B) code-switching-aware supervised fine-tuning of a Qwen2.5-Instruct backbone, and (C) GRPO-guided curriculum RL from high- to mid- to low-resource languages to improve logical correctness and language consistency.</em>
</p>

<p align="center">
  <sub>High-resolution PDF: <a href="figures/cure_med.pdf">Figure 1</a></sub>
</p>







For full details, see the paper (to be added).

## Citation
If you find this work useful, please cite: (to be added)


## Dataset
- **CUREMED-BENCH**: Included in `data.zip`. This dataset features open-ended medical reasoning queries with a single verifiable answer across 13 languages. Unzip it for use in training and evaluation scripts.

