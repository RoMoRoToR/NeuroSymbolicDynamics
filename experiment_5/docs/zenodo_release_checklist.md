# Zenodo Release Checklist

This checklist is the repository-side companion to the GitHub to Zenodo software-archiving flow.

## Repository Preconditions

- `README.md` describes the repository contents and canonical run commands.
- `requirements.txt` lists runtime dependencies.
- `LICENSE` defines redistribution terms.
- `CITATION.cff` provides citation guidance for GitHub.
- `.zenodo.json` provides deposition metadata for Zenodo.
- The repository version in `pyproject.toml` and `CITATION.cff` is updated before each public release.

## GitHub to Zenodo Steps

1. Open Zenodo and connect GitHub under `Profile -> Linked accounts`.
2. Open `GitHub integration` in Zenodo and switch this repository to `On`.
3. Push the release-ready commit to GitHub.
4. In GitHub, open `Releases -> Draft a new release`.
5. Create or select a tag such as `v0.1.0`.
6. Add release notes summarizing the archived software state.
7. Publish the release.
8. Wait for Zenodo to ingest the release and create the software record.
9. Open the Zenodo record, copy the minted DOI, and place it in the manuscript.

## Before Publishing the Release

- Confirm that bundled data files can legally be redistributed in a public archive.
- If you changed authorship, update both `CITATION.cff` and `.zenodo.json`.
- If you changed the release version, update `pyproject.toml` and `CITATION.cff`.
- If Zenodo metadata should differ from GitHub citation guidance, remember that Zenodo will prefer `.zenodo.json`.

## Recommended Code Availability Wording

The source code for the SES software package is available on GitHub and archived on Zenodo with a versioned DOI minted from the corresponding GitHub release. The GitHub repository serves as the active project page, and the Zenodo record serves as the citable archival snapshot used for the manuscript.
