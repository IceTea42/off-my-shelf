# Deploying the web app

The Docker image (see `Dockerfile`) serves the app on port 7860. It runs anywhere Docker runs;
Hugging Face Spaces (Docker SDK) hosts it for free on CPU, which is all the app needs since
every model runs behind the Meta Model API.

## Hugging Face Space

1. Create a Space: **New Space → Docker → Blank**, CPU basic, public.
2. In the Space **Settings → Variables and secrets**:
   - secret `META_API_KEY` = your key
   - variable `OMS_BUDGET_USD` = e.g. `5` (live work stops once this much is spent; the demo
     pack keeps working)
3. Push this repository to the Space with a Space header on its README:

   ```
   git clone https://huggingface.co/spaces/<user>/off-my-shelf space
   git archive HEAD | tar -x -C space          # Windows: use Git Bash for this line
   cd space
   ```

   Add at the very top of `space/README.md`:

   ```
   ---
   title: Off My Shelf
   emoji: 🏷️
   colorFrom: green
   colorTo: gray
   sdk: docker
   app_port: 7860
   license: agpl-3.0
   ---
   ```

   Then `git add -A`, commit and `git push`. The Space builds the image and starts the app.

4. Check `https://<user>-off-my-shelf.hf.space/api/config` returns `"offline": false`.

## Locally with Docker

```
docker build -t off-my-shelf .
docker run --rm -p 7860:7860 --env-file .env off-my-shelf    # http://localhost:7860
```
