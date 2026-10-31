# Start here — Yash's Assignment 3

## What is already prepared

The source code, Colab notebook, data-cleaning pipeline, four model implementations, scoring, temperature experiment, plots/report generators, responsible-AI guide and personalized submission document are included. The CPU basic-Transformer experiment was run and failed to produce useful greedy translations; its real zero scores are retained. Do not confuse this partial run with a completed four-model assignment.

## 1. Get access to the two gated models

1. Open https://huggingface.co and create/sign in to your own account.
2. Open https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and request model access. Review and accept its terms yourself. Approval may not be immediate.
3. Open https://huggingface.co/google/gemma-3-1b-it and review/accept its access conditions yourself.
4. In Hugging Face settings, create a **read** access token. Keep it private. You will enter it only in a hidden login box in Colab. Do not send it in this chat or put it on GitHub.

If you get `401`, `403` or `GatedRepoError`, confirm that the same Hugging Face account has access to both models. There is no code-only workaround for missing approval.

## 2. Open the notebook

1. Download `FGAI_Assignment3_Colab.ipynb` from this package.
2. Open https://colab.research.google.com.
3. Choose **File → Upload notebook** and select that file.
4. Choose **Runtime → Change runtime type → T4 GPU** (or another available GPU) and save.
5. Click the play button on each numbered cell from top to bottom. Read the text above each cell.
6. The notebook mounts Google Drive, installs dependencies, writes all source files and asks for Hugging Face login. You do not have to copy individual Python files.
7. Run all four model cells on the same GPU runtime. Do not reuse CPU timings for a GPU comparison. Progress output and checkpoints are saved in the Drive project directory. A disconnected, unfinished model restarts training when rerun; completed outputs remain.

## 3. Inspect and rate actual translations

1. After the model cells finish, open the notebook's review cell.
2. Download the eight review CSVs or edit them using a spreadsheet editor that preserves UTF-8.
3. For each model, rate at least the first ten test rows in `MODEL_human_review.csv`, then all eight rows in `MODEL_ethical_probes.csv`.
4. Use the scale in `ETHICS.md`. Write short comments describing actual output. A blank translation gets adequacy 0; do not call it safe simply because it contains no offensive text.
5. Upload the edited CSVs using the notebook's review-upload cell. Keep their exact filenames. The finalization check requires ratings and comments.

## 4. Update the existing GitHub repository

1. Open https://github.com/yashshukla27/210911380_FGAI_Assignment3. The repository already exists and contains the prepared project.
2. Keep the implementation, results and report together in this repository.
3. Use this exact repository URL in the finalization cell; no new repository is needed.
4. Paste the URL into the notebook's `REPO_URL` field. Run the final document cell.
5. Run the final download cell. It produces `FGAI_Assignment3_Completed.zip` with code, results, plots, output samples, reports. Extract it on your computer.
6. In the repository, choose **Add file → Upload files**. Upload the extracted project contents, including the `src`, `tests`, `results` and `submission` folders. Do not upload only the ZIP. Avoid uploading `data`, `checkpoints`, token files or cache folders.
7. Commit the upload. Open README, a translation CSV and a plot on GitHub to confirm they are visible. Check access using a signed-out/private browser if the repository is public.
8. Download your executed notebook from Colab and add it to GitHub too, after checking that no token or private information appears in its output. The source-only notebook in the ZIP does not contain execution outputs.

## 5. Submit to LMS

1. Open `submission/Yash_Shukla_210911380_FGAI_Assignment3.docx`. Confirm your details, dataset link and actual repository URL.
2. In Word choose **File → Save As → PDF** if you want the Word-exported version. The notebook also creates a PDF with the same submission details.
3. Make sure the document no longer says **DRAFT** and the repository includes all four measured results and analysis.
4. Upload the final PDF to your course's Assignment 3 LMS section and verify the submission receipt. No LMS upload has been performed by this package.

## Common problems

- **No GPU / CUDA False:** change runtime type before model execution. A GPU is strongly recommended for mT5.
- **Out of memory:** restart the runtime and run one model at a time. mT5 batch size is already one. Clear unused models. If changing dataset/model settings, create a new experiment directory and rerun all comparisons.
- **A model generates empty text or repetitions:** retain it in evaluation; inspect training and validation loss. Improve using validation data (more representative data, a suitable learning rate or more training), then freeze the setup and use a fresh final test set if earlier test results influenced redesign. Never invent a score or remove difficult test examples.
- **Finalization says not ready:** verify that all four keys exist in `results/metrics.json`, the review fields are filled, and `REPO_URL` is a real repository URL.
- **Network/download error:** rerun the affected cell after connectivity returns. Keep the same config and dataset revision.
