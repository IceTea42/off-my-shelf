# Deploying the web app

Every model runs behind an API (Meta Model API, Baseten, Modal), so the app itself needs no GPU.
Two public hosts run the same code: Modal is the main one, the Hugging Face Space the backup.
A new version goes to Modal first; the Space is updated only once Modal has been checked.

## Modal (main host)

1. Once: `uv run modal token new` (opens your browser to log in to Modal).
2. Deploy, from the repo folder, with your keys in `.env`:

   ```
   uv run modal deploy scripts/modal_app.py
   ```

   It prints the app's address (`https://<you>--off-my-shelf-web.modal.run`). The keys in `.env`
   go to Modal as an encrypted secret; runs and the model cache live on the Modal Volume
   `off-my-shelf-data`; live work stops once $10 (list price) is spent.
3. Check it: `<address>/api/config` shows `"offline": false`, then run one photo end to end.
4. The Modal dashboard (modal.com/apps) shows each request, container time and cost;
   `uv run modal volume ls off-my-shelf-data runs` lists the stored runs.

## Hugging Face Space (backup)

The Space `icetea/off-my-shelf` is public, on free ZeroGPU hardware. The Space uses the Gradio
SDK only to run `app.py`, which starts this project's own web app; no Gradio interface is involved.

1. Once: create a token with **write** access at <https://huggingface.co/settings/tokens>, then

   ```
   uv run hf auth login
   ```

2. Publish, once the Modal app is checked (uploads the files tracked by git; `.env`, caches and
   packs never leave your machine):

   ```
   uv run python scripts/deploy_space.py icetea/off-my-shelf --set-secrets --budget 10
   ```

   `--set-secrets` copies the keys set in `.env` (Meta, Baseten, Modal, Logfire) into the
   Space's secrets; `--budget 10` stops live work once $10 (list price) is spent, while the demo
   pack keeps working.

3. The Space builds in a few minutes. Open it, then check `/api/config` shows `"offline": false`.

4. From GitHub instead: add the same write token as an Actions secret named `HF_TOKEN`
   (repo **Settings → Secrets and variables → Actions**), then **Actions → CI → Run workflow** on
   `main`. A push never publishes the Space by itself; secrets set in step 2 stay in place.

## Locally with Docker

```
docker build -t off-my-shelf .
docker run --rm -p 7860:7860 --env-file .env off-my-shelf    # http://localhost:7860
```
