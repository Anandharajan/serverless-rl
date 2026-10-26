# Availability wording for the manuscript

The following wording refers to this public artifact. It does not claim a DOI or journal acceptance. Incorporate it into the actual revised manuscript, checking the journal's placement requirements and the title of the supplementary upload.

## Data availability

The derived per-function workloads, per-function-day evaluation records, paired statistics, trained controller policies and training logs, and AWS calibration measurements supporting this study are publicly available at https://github.com/Anandharajan/serverless-rl (release v1.0.0). The original Azure Functions 2019 dataset is available from Microsoft at https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md under its applicable terms. The Qwen2.5-0.5B-Instruct model is available at https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct; original traces and Qwen model binaries are not redistributed in this repository. Derived workloads include synthetic prompt-length mixtures as documented in the artifact.

## Code availability

The simulator, controller and baselines, workload builder, experiment driver, and AWS calibration and replay-validation code are publicly available at https://github.com/Anandharajan/serverless-rl (release v1.0.0). The repository includes installation instructions, commands for reproducing the statistical analyses and running the experiments, trained controller checkpoints, and artifact checksums. Original code is distributed under the MIT licence. External data, model weights and dependencies retain their respective terms.
