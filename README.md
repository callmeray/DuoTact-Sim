# DuoTact Sim (lightweight mock simulator)

This is a lightweight, non-physical duotact deformation sandbox. The duotact body is modeled as a hollow frame (left/right rails + narrow ridge beam) with only front/back end caps, connected by springs.

## Quick start

1. Install deps (suggest using the existing venv):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the demo to generate and view an interactive Plotly animation in your browser:
   ```bash
   python example.py
   ```

## Files
- `geometry.py` builds the hollow duotact frame (rails + ridge beam), end caps, and spring graph. `ridge_width` controls beam thickness; `ridge_length` lets you shorten the top beam (it stays centered on the body).
- `duotact_simulator.py` supports a mass-spring integrator and a lightweight `static_deform(presses)` that directly applies directional Gaussian displacements (non-physical but stable for quick shape previews).
- `rendering.py` provides Plotly-based 3D visualization (static or animated frames).
- `sensor.py` provides a simple virtual camera (pinhole) to project meshes to an image and save (PNG); used in the example to capture side-face deformation from below.
- `example.py` demonstrates a side press (inward force on a lateral face) using static deformation for stability and saves a camera view (`camera_view.png`) rendered from below.

## Tuning tips
- Increase `nx`/`ny` in `DuotactGeometry` for denser meshes.
- `stiffness`, `spring_damping`, and `global_damping` in `DuotactSimulator` control how stiff and damped the deformation feels.
- Use `static_deform([...])` for quick, stable deformations; each press dict accepts `center`, `direction`, `magnitude`, `radius`. The older `make_press_force` remains for the dynamic path.
