# Deploying the web app

Every model runs behind the Meta Model API, so the app needs no GPU: a free Hugging Face Space
(CPU basic) hosts it. The Space uses the Gradio SDK only to run `app.py`, which starts this
project's own web app; no Gradio interface is involved.

## Modal (main host)

1. Once: `uv run modal token new` (opens your browser to log in to Modal).
2. Deploy, from the repo folder, with your keys in `.env`:

   ```
   uv run modal deploy scripts/modal_app.py
   ```

   It prints the app's address (`https://<you>--off-my-shelf-web.modal.run`). Open it on the phone
   and install it as on the Space (see [phone.md](phone.md)). The keys in `.env` go to Modal as an
   encrypted secret; runs and the model cache live on the Modal Volume `off-my-shelf-data`.
3. The Modal dashboard (modal.com/apps) shows each request, container time and cost.

## Hugging Face Space (backup)

The Space `icetea/off-my-shelf` exists (private, CPU basic). Deploy or update it from the repo:

1. Once: create a token with **write** access at <https://huggingface.co/settings/tokens>, then

   ```
   uv run hf auth login
   ```

2. Each deploy (uploads the files tracked by git; `.env`, caches and packs never leave your
   machine):

   ```
   uv run python scripts/deploy_space.py icetea/off-my-shelf --set-secrets --budget 10
   ```

   `--set-secrets` copies `META_API_KEY` from your `.env` into the Space's secrets;
   `--budget 10` stops live work once $10 (list price) is spent, while the demo pack keeps working.

3. The Space builds in a few minutes. Open it, then check `/api/config` shows `"offline": false`.

4. Automatic updates: add the same write token as a GitHub Actions secret named `HF_TOKEN`
   (repo **Settings → Secrets and variables → Actions**). CI then redeploys the Space after
   every green push to `main`; secrets set in step 2 stay in place.

5. To make it public later: Space **Settings → Space Access control → Public**.

A private Space only opens for you while signed in to Hugging Face, also on the phone.

## Locally with Docker

```
docker build -t off-my-shelf .
docker run --rm -p 7860:7860 --env-file .env off-my-shelf    # http://localhost:7860
```
