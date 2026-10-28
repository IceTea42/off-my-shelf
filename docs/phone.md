# Using Off My Shelf on an Android phone

The web app installs like a phone app (home-screen icon, full screen) and uses the phone's camera
and microphone. Browsers only allow the camera, microphone and install on `https://` pages or on
`localhost`.

## On the public URL

Open the app's `https://` address in Chrome, then **⋮ → Add to Home screen → Install**.
Photos shared from the gallery (**Share → Off My Shelf**) start a run directly.

## On your own computer, through a USB cable

1. On the phone: **Settings → About phone**, tap **MIUI / OS version** 7 times (developer mode),
   then **Settings → Additional settings → Developer options → USB debugging: on**.
2. On the computer, install Android platform-tools:
   `winget install Google.PlatformTools` (Windows) or `sudo apt install adb` (Ubuntu).
3. Plug the phone in, accept the "Allow USB debugging" prompt on the phone, then run:

   ```
   adb devices                      # the phone is listed as "device"
   adb reverse tcp:8000 tcp:8000    # the phone's localhost:8000 is now the computer's
   uv run off-my-shelf serve
   ```

4. On the phone, open Chrome at `http://localhost:8000`, then **⋮ → Add to Home screen → Install**.

`adb reverse` lasts until the cable is unplugged; run it again after reconnecting.
