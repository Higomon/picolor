# picolor

[日本語](README.md) ・ **English**

An experimental system that continuously measures any selected area in a camera image as Lab or Linear RGB, using **12-bit RAW** data from a Raspberry Pi CSI camera.

It is designed for stirred liquids, powders, large objects, coatings, and other samples that are difficult to place inside a conventional color meter.

> [!IMPORTANT]
> picolor is a research prototype, not a spectrometer or a certified colorimeter. Do not use it for medical, safety, regulatory, or trade decisions.

## What it does

| Feature | Description |
|---|---|
| Continuous measurement | A movable `Ref` region observes an 18% gray card while `Target` observes the sample |
| Two color views | Switch between Lab (`L*`, `a*`, `b*`) and Linear RGB |
| 12-bit RAW analysis | Calculate color from signals close to the camera sensor output |
| Color correction | Correct the camera response against the known Lab values of a color reference (48-patch Spyder Checkr) |
| Quality monitoring | Report stability, illumination changes, non-uniformity, clipping, and recalibration needs |
| Recording | Save time-series CSV data, capture settings, warnings, and snapshots |

## 12-bit RAW

When compared with an 8-bit USB-camera output, picolor has 16 times as many theoretical input levels.

| Input | Levels per RAW pixel value | Relative count |
|---|---:|---:|
| 8-bit video | 256 (0–255) | 1× |
| picolor 12-bit RAW | 4096 (0–4095) | 16× |

The current code requests `SBGGR12` from the Raspberry Pi High Quality Camera. It can use fine signal changes before they are rounded to the same 8-bit value.

This does not give `L*` 4096 fixed steps. `L*` is calculated as a floating-point value from the 12-bit RGB input. Practical sensitivity also depends on sensor noise, lighting, exposure, reflections, and calibration.

## Automatic color-reference detection (Spyder Checkr)

- Product used: **Datacolor Spyder Checkr, 48-patch model**
- Form: rigid, folding two-panel case
- Japanese JAN product code: `4571380541088`
- Not supported: SpyderCHECKR 24 or Spyder Checkr Photo/Video

The chart does not need to be perfectly horizontal. picolor automatically detects:

- its position in the image;
- tilt or rotation;
- both panels;
- the centers of all 48 patches.

It maps the patches to the correct Lab references. If position or orientation confidence is insufficient, calibration stops instead of creating an incorrect correction.

## Supported platform

| Item | Current requirement |
|---|---|
| Computer | Raspberry Pi 5 (tested with the 8 GB model) |
| OS | 64-bit Raspberry Pi OS Desktop (Debian 13 Trixie) |
| Camera | High Quality Camera (IMX477) connected through CSI |
| RAW input | 12-bit `SBGGR12` |
| Camera API | Picamera2/libcamera |
| Not supported | Windows, macOS, USB cameras, or other CSI cameras |

**A 12-bit RAW CSI camera and the Raspberry Pi camera stack are required. Windows and macOS are not supported.**

> [!NOTE]
> The current release is Raspberry Pi-specific. A typical Windows PC or Mac has neither a direct connector for a Raspberry Pi CSI camera nor the Picamera2 environment. Some embedded systems provide MIPI CSI-2, but different connectors, drivers, and RAW formats mean that picolor will not run on them without modification.

## Required hardware

| Category | Equipment | Notes |
|---|---|---|
| Raspberry Pi | Raspberry Pi 5, microSD 32 GB or larger, stable USB-C supply, case and cooling | Tested with the 8 GB model and official 27 W supply |
| Camera | Raspberry Pi High Quality Camera, IMX477 C/CS-mount | Target of the current code |
| Lens | Raspberry Pi 6 mm wide-angle CS-mount lens | The 16 mm lens is also an option |
| Camera link | Standard–Mini cable, two Pimoroni CSI–HDMI adapter boards, standard HDMI cable, short 15-pin CSI cable | Petit Studios adapter kit; now retired |
| Color references | Color-reference chart (48-patch Spyder Checkr) and 18% gray card | Gin-ichi Silk Gray Card Ver.2 used in the tested system |
| Capture setup (small samples) | [HAKUBA LED Studio Box 60 (AMZLEDSBX60)](https://www.amazon.co.jp/dp/B0923V3439), fixed mounts | Approx. 64×62×63 cm; includes LEDs and white, black, and orange backgrounds |
| Capture setup (large samples) | White LEDs, diffuser or softboxes, white background, fixed mounts | Keep camera, lighting, and sample fixed |
| Calibration and control | Lens cap, HDMI monitor, keyboard, mouse | Used for setup and operation |

### Camera connection

```text
Raspberry Pi 5 CAM/DISP
  → Standard–Mini camera cable
  → CSI–HDMI adapter board
  → standard HDMI cable
  → CSI–HDMI adapter board
  → 15-pin CSI camera cable
  → High Quality Camera
```

> [!CAUTION]
> The HDMI cable carries the extended camera signal. Never connect it to a Raspberry Pi display output or a monitor. Power off the Pi before changing camera wiring. Use a short HDMI cable with the required signal and shield connections.

## Setup

The tested OS is 64-bit Raspberry Pi OS Desktop based on Debian 13 Trixie.

```bash
sudo apt update
sudo apt install -y \
  git python3 python3-numpy python3-opencv python3-pil \
  python3-picamera2 python3-scipy fonts-noto-cjk
```

Check the camera:

```bash
rpicam-hello --timeout 5000
```

Clone and start picolor:

```bash
git clone https://github.com/Higomon/picolor.git
cd picolor
python3 -u -c "from csi.main import main; main()"
```

## First calibration

Follow the on-screen guide in this order.

| Step | Key | Place in view | Purpose |
|---:|:---:|---|---|
| 1 | `D` | Lens cap | Dark-noise correction |
| 2 | `F` | Uniform white background | Illumination non-uniformity correction |
| 3 | `W` | 18% gray card | White balance and relative reference |
| 4 | `P` | Open color reference (48-patch Spyder Checkr) | 48-patch color correction |

The color reference (Spyder Checkr) need not be horizontal, but keep the full chart visible without reflections or shadows covering its patches. Detection may fail if the chart is far sideways, outside the frame, strongly reflected, or heavily shadowed.

## Measurement

1. Align `Ref` with the 18% gray card.
2. Align `Target` with the sample area.
3. Wait for the measurement-ready indication.
4. Press `m` to start recording and `s` to stop.
5. Press `q` or `ESC` to quit.

<details>
<summary><strong>Suitable samples and considerations</strong></summary>

| Sample | Use | Consideration |
|---|---|---|
| Stirred liquid | Follow reactions or color adjustment | Keep bubbles, vessel, and reflections consistent |
| Powder or granules | Compare surface color | Keep thickness, packing, and shadows consistent |
| Large object | Continuously measure one area | Fix distance, angle, and ambient light |
| Coating or print | Compare selected locations | Control gloss and reflection direction |
| Changing sample | Record drying, fading, or reaction | Do not move the camera or reference |

</details>

<details>
<summary><strong>Controls and saved data</strong></summary>

| Key | Action |
|:---:|---|
| `D` / `F` / `W` / `P` | Dark / flat / gray-card / 48-patch calibration |
| `V` | Verify the gray-card state |
| `Tab` | Switch Lab / Linear RGB |
| `m` / `s` | Start / stop continuous recording |
| `c` | Recalibrate after a setup change |
| `q` / `ESC` | Quit |

Saved data can include time-series CSV, timestamps, exposure, gain, reference state, warnings, snapshots, and calibration records.

- Calibration: `calibration/`
- Results: `/home/<user>/picolor/results/`

</details>

<details>
<summary><strong>Measurement method and limitations</strong></summary>

- `Ref` and `Target` are measured in the same image to reduce illumination drift.
- The color reference (Spyder Checkr) provides known Lab values as the absolute reference; the 18% gray card is the relative in-frame reference.
- Dark noise, illumination non-uniformity, and white balance are also corrected.
- picolor does not measure wavelength spectra or UV-Vis absorbance.
- Gloss, bubbles, shadows, ambient light, distance, and angle affect results.
- Recalibrate after changing the camera, lighting, distance, aperture, or reference placement.
- Accuracy equivalent to a commercial colorimeter is not guaranteed. Validate with known samples for each intended use.

</details>

<details>
<summary><strong>Technical references</strong></summary>

- [Raspberry Pi Camera documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [High Quality Camera](https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/)
- [RAW mode and bit depth](https://www.raspberrypi.com/documentation/computers/camera_software.html#mode)
- [Camera Cable](https://www.raspberrypi.com/products/camera-cable/)
- [Pimoroni CSI–HDMI extension](https://shop.pimoroni.com/products/pi-camera-hdmi-cable-extension)
- [Datacolor Spyder Checkr](https://www.datacolor.com/spyder/products/spyder-checkr/)

</details>

## License

[MIT License](LICENSE)

## Status

picolor is under active experimental development. Raspberry Pi OS or hardware updates may change its behavior.

Raspberry Pi, Datacolor, and SpyderCHECKR are trademarks of their respective owners. This project is not an official product of those companies.
