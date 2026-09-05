# Rendered diagram exports

Mermaid sources live in [`../diagrams/`](../diagrams/). GitHub renders those blocks
inline in [`../architecture.md`](../architecture.md), so nothing here is needed to
read the documentation.

Export to this folder only when a diagram is going into a slide deck, a PDF, or
anywhere Mermaid will not render.

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i ../diagrams/shared-pipeline.mmd     -o shared-pipeline.png     -w 1600
mmdc -i ../diagrams/hw3-ensemble.mmd        -o hw3-ensemble.png        -w 1600
mmdc -i ../diagrams/asl-tflite-pipeline.mmd -o asl-tflite-pipeline.png -w 1600
```

Regenerate whenever the `.mmd` source changes. Never edit an exported image by hand —
the `.mmd` file is the source of truth.
