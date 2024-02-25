# One Prompt To Rule Them All: LLMs for Opinion Summary Evaluation

Paper Link: [One Prompt To Rule Them All: LLMs for Opinion Summary Evaluation](https://arxiv.org/abs/2402.11683)

## Abstract
Evaluation of opinion summaries using conventional reference-based metrics rarely provides a holistic evaluation and has been shown to have a relatively low correlation with human judgments. Recent studies suggest using Large Language Models (LLMs) as reference-free metrics for NLG evaluation, however, they remain unexplored for opinion summary evaluation. Moreover, limited opinion summary evaluation datasets inhibit progress. To address this, we release the SUMMEVAL-OP dataset covering 7 dimensions related to the evaluation of opinion summaries: fluency, coherence, relevance, faithfulness, aspect coverage, sentiment consistency, and specificity. We investigate Op-I-Prompt a dimension-independent prompt, and Op-Prompts, a dimension-dependent set of prompts for opinion summary evaluation. Experiments indicate that Op-I-Prompt emerges as a good alternative for evaluating opinion summaries achieving an average Spearman correlation of 0.70 with humans, outperforming all previous approaches. To the best of our knowledge, we are the first to investigate LLMs as evaluators on both closed-source and open-source models in the opinion summarization domain.

## Citation
```
@misc{siledar2024prompt,
      title={One Prompt To Rule Them All: LLMs for Opinion Summary Evaluation}, 
      author={Tejpalsingh Siledar and Swaroop Nath and Sankara Sri Raghava Ravindra Muddu and Rupasai Rangaraju and Swaprava Nath and Pushpak Bhattacharyya and Suman Banerjee and Amey Patil and Sudhanshu Shekhar Singh and Muthusamy Chelliah and Nikesh Garera},
      year={2024},
      eprint={2402.11683},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```