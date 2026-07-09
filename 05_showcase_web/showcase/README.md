# 3DGS Showcase

Start a local static server from the bundle root, then open the page through HTTP:

```bash
scripts/start_showcase.sh
```

Then open `http://127.0.0.1:8765/showcase/index.html`.

Do not use `file://` for the interactive viewer, because it uses Three.js ES modules.

The page expects this layout:

- `showcase/index.html`
- `outputs/...`

It embeds final MP4 demos, links to exported PLY/OBJ assets, and loads compact interactive assets for the Three.js viewer. Large raw render frame directories are intentionally not required.
