---
title: CropGuard AI
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
short_description: Real PlantVillage-trained crop disease classification with Grad-CAM and class-specific advisory.
---

# CropGuard AI

This Hugging Face Space runs the CropGuard application from the GitHub source of truth.

The Docker build downloads the existing trained PlantVillage checkpoint and runtime files from the repository at build time. No retraining or model conversion is performed by the Space deployment.
