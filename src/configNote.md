# Use environment variables to configure the settings

## RAG Configuration

 NOTE: Changed from hardcoded values (e.g., CHUNK_SIZE = 1024) to environment variable pattern
 (e.g., CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))) in Phase 1.

 in this file we use environment variables to configure the settings. and the second value is the default value if the environment variable is not set.
 for example, CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024")) means that the default value is 1024 if the environment variable is not set.

 Why this change:

- Hardcoded: Fixed values in code, requires code edit to change settings
- Environment variable: Read from .env file or system environment, allows runtime configuration

 Benefits:

- Different settings for different hardware (laptop vs. EVO-X2) without code changes
- Easy experimentation during benchmarking (change .env, no code restart needed)
- Deployment flexibility (dev/staging/prod configs via environment, not code)
- Same codebase works across all environments

***Just edit .env file, don't touch config.py***

## Supported document formats (via document_loaders.py)

.txt, .md, .pdf, .docx, .csv, .json, .html, .htm, .xlsx, .xls
