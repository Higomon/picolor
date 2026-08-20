# picolor

[日本語README](README.md)

**picolor** is an experimental color-measurement system built with a Raspberry Pi and a CSI camera. It continuously measures the color of a selected area in the camera image.

It is intended for samples that are difficult to place inside a conventional color meter or spectroscopic instrument, including:

- liquids whose color changes while being stirred;
- large objects or parts of installed equipment;
- powders, granules, sheets, coatings, and printed materials;
- samples that change during a reaction, drying, or fading process.

> [!IMPORTANT]
> picolor is a research prototype. It is not a spectrometer or a certified commercial colorimeter. Do not use it for medical decisions, safety decisions, regulatory compliance, or trade certification.

## What it can do

picolor displays two movable measurement regions:

- **Ref** observes a stable reference such as an 18% gray card.
- **Target** observes the sample.

Because both regions are captured in the same image, picolor can reduce the influence of small changes in illumination.

It continuously displays and records:

- **CIELAB** values: `L*` for lightness, `a*` for green–red, and `b*` for blue–yellow;
- **Linear RGB** values from the camera signal;
- measurement stability and illumination warnings;
- exposure settings, reference drift, and quality information;
- time-series CSV data and measurement snapshots.

The display can be switched between Lab and Linear RGB.

## Color references and calibration

picolor uses two physical references:

1. **Datacolor SpyderCHECKR 48**
   - Its 48 known color patches are used to correct the camera's color response.
   - picolor detects the chart position and orientation and reads all patches.
2. **18% gray card**
   - It provides a stable in-frame reference during measurement.
   - It is used to monitor changes in brightness and color.

The system also calibrates camera dark noise, illumination non-uniformity, and white balance.

## Suitable samples

| Sample or situation | Example | Important consideration |
|---|---|---|
| Stirred liquid | Follow a color change in a vessel | Keep reflections, bubbles, and the vessel consistent |
| Powder or granules | Measure a level surface in a tray | Keep thickness, packing, and shadows consistent |
| Large object | Measure one fixed area | Fix camera distance, angle, and lighting |
| Coating, print, or sheet | Compare selected locations | Control gloss and reflection direction |
| Time-dependent sample | Record reaction, drying, or fading | Do not move the camera or reference |

## Required hardware

### Core equipment

- **Raspberry Pi 5** — tested with the 8 GB model
- **Raspberry Pi High Quality Camera** — IMX477, C/CS-mount version
- **Raspberry Pi recommended 6 mm wide-angle CS-mount lens**
  - The recommended 16 mm lens is an option for a narrower field of view.
- **Short Raspberry Pi Camera Cable Standard–Mini**
  - Connects the Pi 5's 22-pin camera connector to the first 15-pin CSI–HDMI adapter board.
- **Raspberry Pi Camera HDMI Cable Extension formerly sold by Pimoroni**
  - A two-board CSI–HDMI extension kit made by Petit Studios.
  - This setup uses a robust HDMI cable instead of a long flat CSI ribbon.
- **Standard HDMI cable**
  - Connects the two CSI–HDMI adapter boards. Use a short, good-quality cable with all required signal and shield connections.
- **Short 15-pin CSI camera cable**
  - Connects the camera-side adapter board to the HQ Camera.
- **Stable USB-C power supply** — official Raspberry Pi 27 W supply or equivalent 5 V / 5 A supply
- **microSD card, 32 GB or larger**, with 64-bit Raspberry Pi OS
- **Datacolor SpyderCHECKR 48**
- **18% gray card**
- **Stable white LED lighting**
- **Light tent, diffuser, or softbox**
- **Tripod, mounting arm, or fixed camera fixture**
- **Uniform white background**
- **Lens cap** for dark calibration
- **HDMI monitor, keyboard, and mouse** for setup and operation

### Choose according to sample size

- Small objects: an LED light tent around 60 cm wide
- Large objects: two LED lights, two softboxes, and two light stands
- Long operation: a Raspberry Pi case with a fan or Active Cooler
- Reproducible work: a fixture that keeps the camera, gray card, and sample in the same positions

Stable geometry and lighting matter more than expensive equipment.

Official references:

- [Raspberry Pi Camera documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [Raspberry Pi Camera Cable](https://www.raspberrypi.com/products/camera-cable/)
- [Pimoroni: Raspberry Pi Camera HDMI Cable Extension](https://shop.pimoroni.com/products/pi-camera-hdmi-cable-extension) (retired product)
- [Datacolor SpyderCHECKR](https://www.datacolor.com/spyder/products/spyder-checkr/)

### Camera cable connection

The tested system is connected as follows:

```text
Raspberry Pi 5 CAM/DISP connector
  → short Standard–Mini camera cable
  → CSI–HDMI adapter board
  → standard HDMI cable
  → CSI–HDMI adapter board
  → short 15-pin CSI camera cable
  → Raspberry Pi High Quality Camera
```

> [!CAUTION]
> In this setup, HDMI cable is only used as wiring for the extended camera signal. Never connect this cable to the Raspberry Pi's micro-HDMI display output or to a monitor. Power off the Raspberry Pi before connecting or disconnecting any camera cable. Pimoroni's former product page notes that an HDMI cable without the required data shielding may prevent camera detection.

## Software setup

The 64-bit Raspberry Pi OS Desktop edition is recommended.

```bash
sudo apt update
sudo apt install -y \
  git \
  python3 \
  python3-numpy \
  python3-opencv \
  python3-pil \
  python3-picamera2 \
  python3-scipy \
  fonts-noto-cjk
```

Connect the camera and both adapter boards as shown above while the Raspberry Pi is powered off. Use the Standard–Mini cable between the Pi 5 and the first adapter board.

Test the camera:

```bash
rpicam-hello --timeout 5000
```

Clone and start picolor:

```bash
git clone https://github.com/Higomon/picolor.git
cd picolor
python3 -u -c "from csi.main import main; main()"
```

Do not begin production measurements before calibration.

## First calibration

Follow the on-screen instructions in this order:

1. Fit the lens cap and press `D` to capture dark noise.
2. Remove the cap and all samples, show only a uniform white background, and press `F` to correct illumination non-uniformity.
3. Place the 18% gray card and press `W` to set white balance and the measurement reference.
4. Place the SpyderCHECKR 48 and press `P` to correct the camera's color response.
5. Remove the chart. Put the gray card inside the `Ref` region and the sample inside the `Target` region.

Wait until the display reports that measurement is allowed.

## Measurement

1. Align `Ref` with the 18% gray card.
2. Align `Target` with the area of the sample to measure.
3. Wait for stable values and a measurement-ready indication.
4. Press `m` and enter an output name.
5. Press `s` to stop recording.
6. Press `q` or `ESC` to quit.

| Key | Action |
|---|---|
| `D` | Dark-noise calibration |
| `F` | Illumination non-uniformity calibration |
| `W` | Gray-card reference and white balance |
| `P` | SpyderCHECKR 48 color calibration |
| `V` | Verify the gray-card state |
| `Tab` | Switch Lab / Linear RGB display |
| `m` | Start continuous recording |
| `s` | Stop continuous recording |
| `c` | Start recalibration after a condition change |
| `q` / `ESC` | Quit |

## Saved data

picolor can save:

- Lab or Linear RGB time-series CSV files;
- timestamps, exposure time, and gain;
- reference brightness, color, and uniformity;
- stability and warning states;
- snapshots with measurement values;
- dated calibration and acceptance records.

Calibration files are stored under `calibration/`. Measurement results are stored under `/home/<user>/picolor/results/` on Raspberry Pi. These files can contain experiment details or sample names and should not be committed to GitHub.

## Limitations

- picolor does not measure a wavelength spectrum.
- It cannot measure a UV-Vis absorbance spectrum.
- Accurate transmission measurements require a controlled cell, optical path, and background that are not included here.
- Gloss, reflections, bubbles, shadows, and ambient light can strongly affect results.
- Recalibrate after changing the camera, lighting, distance, angle, aperture, or reference placement.
- Color charts and gray cards can age, fade, or become dirty.
- Accuracy equivalent to a commercial colorimeter is not guaranteed. Validate the system with known samples for each intended application.

## Source layout

```text
csi/
├── main.py                 # Application entry and processing loop
├── camera.py               # CSI camera and RAW capture
├── colorimeter_common.py   # Color calculation, calibration, gates, and storage
├── key_handler.py          # Keyboard controls and calibration workflow
├── overlay.py              # On-screen interface
├── logger.py               # CSV logging
├── flat_2d_smoothed.py     # Illumination non-uniformity correction
├── flat_radial_profile.py  # Radial shading helper
└── key_commands.py         # Key decisions
```

This public repository does not include SSH configuration, host addresses, passwords, private deployment scripts, calibration files, or experimental data.

## Project status

picolor is under active experimental development. Behavior may change with hardware or Raspberry Pi OS updates.

## License

No reuse license is currently granted. Contact the repository owner before using, modifying, or redistributing the code.

## Trademarks

Raspberry Pi is a trademark of Raspberry Pi Ltd. Datacolor and SpyderCHECKR are trademarks of their respective owners. This is an independent project and is not an official product or warranty from those companies.
