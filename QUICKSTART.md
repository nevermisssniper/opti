# OPTI — runnable workbench

Explore how a fitted model responds to behavioral inputs, with historical context fixed.

## Open the interface

Python 3.10+; no npm install, API keys, model download or GPU required for the demo.

```sh
python serve_demo.py
```

Open <http://127.0.0.1:8000>. To use another port: `python serve_demo.py --port 8001`.
Serve `demo/` over HTTP rather than opening `index.html` as a file. The server binds only to localhost.

## What is implemented

The browser uses exported Ridge weights from the Python pipeline: 157 training rows and 50 later evaluation rows generated with seed 42. Historical lag values remain fixed in what-if scenarios. Cross-language prediction parity is tested. Interactions make the constrained objective potentially nonconvex; SLSQP returns a local candidate, not a guaranteed global optimum.

Interface capabilities: What-if sliders, Coefficient view, Held-out metrics.

## Core workflow

From the repository root (install the project's existing dependencies first for model-backed commands):

```sh
python -m opti.export_demo --output demo/model.json
```

## Verify

The browser's calculation tests use Node 18+ and no third-party packages:

```sh
node --test demo/core.test.cjs
python -m pytest tests -q
```

The GitHub demo workflow checks browser calculations and the dependency-free core tests where present. Full original model/service suites have separate dependencies; passing demo checks does not certify GPU training or external services.

## Before live use

These synthetic associations are not personal predictions or causal recommendations.

Local browser data belongs to this browser profile. Export records before clearing it or changing devices. Photo previews stay in memory; exported briefs contain filenames, not uploaded image bytes. There are no payment or outbound model calls in this demo.

## Provenance

Workbench and starter additions: AI-assisted implementation, 2026-11-13, for Ray's review. Existing repository history and upstream attribution are preserved. New starter directions are proposals where the GitHub repository had no code.
