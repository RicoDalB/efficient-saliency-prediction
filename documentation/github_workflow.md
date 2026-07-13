# GitHub workflow

## Storage separation

Google Drive is used for:

- dataset;
- checkpoints;
- predictions;
- experiment outputs;
- backups.

GitHub is used for:

- notebooks;
- split definitions;
- documentation;
- report material.

## Beginning of a Colab session

1. Mount Google Drive.
2. Clone or pull this repository.
3. Load the dataset from Drive.
4. Work on one specific experiment.

## End of a Colab session

1. Save the notebook to Drive.
2. Copy the updated notebook into the Git repository.
3. Review `git status`.
4. Commit the changes.
5. Push to GitHub.

## Files that must never be committed

- SALICON data;
- checkpoints;
- large prediction folders;
- credentials;
- access tokens.
