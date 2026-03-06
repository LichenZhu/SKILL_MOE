---
name: action_storyboard
description: Kinematic Storyboard Prompting for motion/action questions. Extracts 4 equidistant frames, detects the highest-motion region via frame difference, applies Spotlight encoding (darken + red box), composes a 2×2 grid, and overlays red trajectory arrows showing object movement across frames.
tags: action, motion, trajectory, storyboard, visual-evidence, frame-difference
when_to_use: Questions about motion direction, movement trajectory, action sequence, which direction something moves, approaching/retreating, or continuous actions where temporal progression matters (e.g. "which direction does X move", "how does Y approach Z", "what is the trajectory of").
skill_type: support
---

## How it works

Extracts 4 keyframes from the relevant temporal window, detects the highest-motion
region in each via frame difference (not optical flow), applies Spotlight encoding
(darken surrounding + red bounding box), composes a 2×2 grid, and draws red
trajectory arrows showing the spatial path of the main moving object across time.
Output is stored as `visual_evidence` and injected directly into the VLM.

## Output
- **Type**: support — provides a 2×2 storyboard grid as visual evidence; does not output option_scores.
- `artifacts["visual_evidence"]`: list containing one base64 JPEG storyboard grid.
