"""Allow running Prism as `python -m prism`."""

import logging
import os
import warnings

# Suppress noisy library warnings before anything imports
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

from prism.cli.app import main

if __name__ == "__main__":
    main()
