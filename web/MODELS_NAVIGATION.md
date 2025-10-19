# VMEvalKit Web Dashboard - Model Navigation Guide

## 🎉 All Models Are Now Accessible!

The web dashboard has been updated to make all video generation models easily accessible and navigable.

## What Was Fixed

Previously, only the Sora model content was easily visible. The issue was that all model sections were collapsed by default with no clear indication that they were clickable.

## New Features Added

### 1. **Auto-Expand First Model**
- The first model in the list now automatically expands when you load the page
- This gives immediate visibility that there's content available

### 2. **Visual Indicators**
- Model and domain headers now have:
  - Cursor pointer on hover
  - Background color changes on hover
  - "Click to expand/collapse" text indicators
  - Arrow icons (▶/▼) showing the current state

### 3. **Quick Navigation Bar**
- A yellow navigation bar at the top with buttons for each model
- Click any model button to jump directly to that model's section
- The section automatically expands when you navigate to it

### 4. **Expand/Collapse All Buttons**
- "Expand All Models" button - opens all model sections at once
- "Collapse All Models" button - closes all model sections at once

## Available Models (6 total)

All of these models are now accessible in the dashboard:

1. **openai-sora-2** - OpenAI's Sora model
2. **wavespeed-wan-2.2-i2v-720p** - Wavespeed image-to-video model
3. **veo-3.1-720p** - Google's Veo 3.1 model
4. **veo-3.0-generate** - Google's Veo 3.0 model  
5. **luma-ray-2** - Luma's Ray model
6. **runway-gen4-turbo** - Runway's Gen-4 Turbo model

## How to Access the Dashboard

The web server is running at: http://localhost:5000

Simply open this URL in your browser to see all the model results.

## Data Summary

- **Total Models**: 6
- **Total Domains**: 5 (chess, maze, raven, rotation, sudoku)
- **Total Inferences**: 450 (75 per model)

Each model has been tested on all 5 reasoning domains with 15 tasks per domain.
