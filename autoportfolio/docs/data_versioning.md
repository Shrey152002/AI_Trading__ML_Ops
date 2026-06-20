# Data Versioning with DVC

AutoPortfolio uses [DVC](https://dvc.org) to version the two dataset directories that the
pipeline generates: raw ingested OHLCV data (`data/raw/`) and versioned engineered features
(`data/feature_store/`). Git tracks small pointer files (`*.dvc`); the actual data bytes live
in a local, gitignored DVC remote at `../.dvc-storage/` (sibling to `autoportfolio/`).

This was verified end-to-end during development: real banking-portfolio data was ingested,
a feature_store version was computed, `dvc add` + `dvc push` ran successfully, and
`git status` showed only the small `.dvc` pointer files as new — not the underlying parquet.

## Why a manual step, not part of the nightly pipeline

`scheduler/pipeline.py` does **not** call `dvc add`/`dvc push` automatically. Snapshotting a
dataset version is a deliberate decision — it changes what `git log` on `*.dvc` files means
("this is the dataset version model vX was trained on") — and mutating git/DVC state from an
unattended nightly job is a meaningfully different risk than just having the capability
available. Run it yourself (or wire it into CI) when you want to pin a version.

## Workflow

After a pipeline run produces new data for a portfolio:

```bash
# from autoportfolio/
dvc add data/raw/<portfolio> data/feature_store/<portfolio>
git add data/raw/<portfolio>.dvc data/raw/.gitignore \
        data/feature_store/<portfolio>.dvc data/feature_store/.gitignore
git commit -m "Snapshot <portfolio> dataset as of <date>"
dvc push   # uploads the actual bytes to the local remote
```

To reproduce a past version on another machine (or after `git checkout` of an older commit):

```bash
git checkout <commit>   # restores the .dvc pointer files to that point in time
dvc pull                # downloads the matching data bytes from the remote
```

## Remote configuration

The remote is configured in `.dvc/config` (committed to git — it's just a path, not
credentials):

```bash
dvc remote add -d localstorage ../.dvc-storage
```

To switch to a shared/cloud remote later (e.g. S3), only this one remote definition needs to
change — `dvc remote modify localstorage url s3://your-bucket/autoportfolio` plus AWS
credentials via the usual DVC/boto mechanisms. No application code depends on where the remote
physically lives.

## What's intentionally not versioned

`data/validated/` (validation reports) and `reports/*.html`/`*.json` are regenerated outputs,
not source datasets — they stay in `.gitignore` rather than DVC. `data/features/` is the legacy
standalone-CLI cache from `data/features.py`'s own save path (`python -m data.features
<portfolio>`); the canonical, versioned feature data is `data/feature_store/`.
