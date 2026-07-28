# Cloudflare Tunnel for the FPL Robo dashboard

Exposes `http://127.0.0.1:8787` over HTTPS without opening router ports.

## Recommended on this Pi: long-lived Quick Tunnel in tmux

No Cloudflare domain required. Keep **one** `cloudflared` process running so the
`*.trycloudflare.com` URL stays the same (it only changes if that process dies).

```bash
# Ensure systemd quick-tunnel is off (it restarts → new URL)
sudo systemctl disable --now cloudflared-fpl-robo-quick.service

chmod +x /home/callum/Documents/FPL_robo/deploy/start-cloudflare-tmux.sh
/home/callum/Documents/FPL_robo/deploy/start-cloudflare-tmux.sh
cat /home/callum/Documents/FPL_robo/data/cloudflare_tunnel_url.txt
tmux attach -t fpl-tunnel   # Ctrl-b d to detach without killing
```

Do **not** restart the tmux session unless you accept a new URL.

For a hostname that survives process restarts, use a named tunnel + domain below.

## Stable hostname (named tunnel)

Needs a domain on Cloudflare. Prefer this if you want one permanent URL.

### 1. Cloudflare account

Sign up / log in at https://dash.cloudflare.com (free is fine).

### 2. Login on the Pi (opens a browser URL)

```bash
cloudflared tunnel login
```

Open the printed URL, pick any Cloudflare zone (or create a free one). This writes certs under `~/.cloudflared/`.

### 3. Create a named tunnel

```bash
cloudflared tunnel create fpl-robo
```

Note the tunnel UUID it prints. Then:

```bash
cp /home/callum/Documents/FPL_robo/deploy/cloudflared-config.example.yml ~/.cloudflared/config.yml
```

Edit `~/.cloudflared/config.yml` and replace `YOUR_TUNNEL_UUID` with that UUID (and the credentials file name if different).

### 4. Public hostname

```bash
cloudflared tunnel route dns fpl-robo fpl.yourdomain.com
```

Set `hostname: fpl.yourdomain.com` under `ingress` in `config.yml`.

Disable the quick-tunnel unit first so only one tunnel runs:

```bash
sudo systemctl disable --now cloudflared-fpl-robo-quick.service
sudo cp /home/callum/Documents/FPL_robo/deploy/cloudflared-fpl-robo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-fpl-robo.service
```

## Optional: Cloudflare Access (login gate)

Zero Trust → Access → Application → protect the hostname so only invited emails can open it. Recommended if the link is shared.

## Uninstall named tunnel

```bash
sudo systemctl disable --now cloudflared-fpl-robo.service
cloudflared tunnel delete fpl-robo   # after removing DNS route
```
