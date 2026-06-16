# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-06-11

### Added
- Initial release of seamless core package
- Kinematics computation module (core): compute derivatives, error metrics, kinematics analysis
- Neural flow networks module (flow): FlowMLP, SceneFlowMLP, UVFlowMLP with training pipelines
- Surface parameterization module (cartography): NuvoMLP for multi-chart surface mapping
- Visualization utilities module (vis): napari, matplotlib, and plotly integration
- Data I/O utilities module (utils): loading and saving for various formats
- Synthetic data generation module (synth): procedural geometry and topology generation
- Optimization module (optim): training routines for parameterization networks
- Network architectures module (networks): mapping MLP implementations
- Comprehensive type hints across all modules
- Full test suite with backward compatibility validation
