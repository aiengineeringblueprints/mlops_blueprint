# Blueprint documentation

The MLOps blueprint consists of a reference architecture and a reference implementation for two
parts of the ML lifecycle:

| Part | Architecture | Implementation | Usage |
|---|---|---|---|
| Training: data versioning, experiment tracking, champion selection, model registry | [training-architecture.md](training-architecture.md) | `model_training/` | `model_training/README.md` |
| Deployment: packaging, serving and monitoring a registered model | [deployment-architecture.md](deployment-architecture.md) | `model_deployment/` | `model_deployment/README.md` |

The architecture documents state the requirements each part addresses, its components and
workflow, the design decisions with their rationale, and known limitations. They are derived from
the theses the blueprint is based on (Saur 2026, training; Kachhadiya 2026, deployment);
"Implementation note" lines mark where the implementation deviates from or extends them.

Paths are relative to the root of the blueprint, the directory that contains `model_training/`,
`model_deployment/` and this `docs/` directory.

The figures in `figures/` are the original figures of the two theses, reproduced unchanged with their
source; captions and implementation notes explain where the current implementation differs.
