# Data, dependencies, and licensing

Author: Anandharajan

The root MIT licence covers Anandharajan's code and original documentation. It does not override upstream data/model licences, software dependency licences or third-party rights in derived data. The empirical records are provided for research inspection and reproduction; consult upstream terms before redistributing derived workloads.

* Azure Functions 2019 source and download instructions: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md . Only derived per-function workloads and selection metadata are included here. The long-prompt fraction is synthetic.
* Qwen model source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct . Calibration used quantized GGUF representations (Q8_0 primary and Q3_K_M fallback). Model binaries are not included. The exact upstream GGUF revision and binary hashes were not recorded in this artifact, which limits bit-for-bit live repetition.
* Dependencies retain their own licences. Offline simulation needs NumPy, pandas, SciPy and PyTorch. Optional AWS measurement code additionally needs boto3 and llama-cpp-python in a compatible Lambda runtime.
* Released `.pt` files contain this project's trained controller state dictionaries, not Qwen weights. Training logs accompany those policies. Only load checkpoints from trusted sources.

`MANIFEST.sha256` records the release inputs (code, workloads, measurements and results). `requirements-verified.txt` records direct package versions used to verify this release; it is not a claim about the original training environment. The historical AWS REPORT log, discarded first campaign, private correspondence, deployment credentials and original trace/model downloads are not published.
