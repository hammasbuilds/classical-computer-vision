# 11 · HDR exposure fusion — classical computer vision

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV 4.14](https://img.shields.io/badge/OpenCV-4.14-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![No deep learning](https://img.shields.io/badge/deep%20learning-none-success)](#)
[![Runs on CPU](https://img.shields.io/badge/hardware-CPU%20only-lightgrey)](#)

A single exposure cannot hold a sunlit window and the dark room around it.
Bracketing takes several; fusion combines them. Every write-up shows the result
and declares it better.

**Better than what, by how much, and against what ceiling?**

**No neural network, no training, no GPU.**

---

## Results

Four scenes of **12 stops** — far wider than 8 bits — bracketed into five
exposures two stops apart and fused. The reference is not a matter of taste: it
is a fixed tone curve applied to the **exact radiance**, so every method is
being asked *how close did you get to what you would have produced with perfect
information*.

![Four scenes, six methods](docs/images/compare_fusion.png)

| Sr | Scene | Mertens | Debevec + Reinhard | Debevec + Drago | Debevec + Mantiuk | **Mean of frames** | **Middle exposure** |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | lake and shrine · bright sky, dark foreground | 0.873 | 0.735 | 0.817 | 0.623 | **0.943** | 0.896 |
| 2 | woman and child · close faces, flat light | 0.835 | 0.562 | 0.647 | 0.464 | **0.932** | 0.897 |
| 3 | giraffe · lit animal, flat background | 0.874 | 0.676 | 0.731 | 0.586 | **0.931** | 0.887 |
| 4 | harbour and boat · sunlit town, shaded hull | 0.870 | 0.619 | 0.594 | 0.479 | **0.931** | 0.892 |

Cells are SSIM against the oracle. **The two bold columns are the controls.**

> **Every real fusion method loses to averaging the frames.** The naive mean —
> which gives a blown-out white pixel exactly the same vote as a correctly
> exposed one — scores **0.928 SSIM** against Mertens' **0.848** and
> Debevec + Reinhard's **0.634**. It also beats them on PSNR, 23.5 dB to 20.9,
> and runs in 9 ms against Debevec's 1.8 s.

> **Taking one photograph and doing nothing beats all three Debevec pipelines.**
> The middle exposure scores **0.890 SSIM** in 0.002 ms. Every radiance-based
> pipeline scores lower and takes between 3,500× and 900,000× longer.

**Part of that is real and part of it is the metric, and the difference matters.**
The oracle renders with a *global* Reinhard curve, so a method that behaves
globally matches it more closely than one that blends locally — which is exactly
what Mertens does, and exactly what a human would prefer looking at the picture.
The honest statement is not "averaging is the best HDR method". It is:

- against a global reference, global methods win, and the reference here is global;
- **the sophisticated pipelines are not losing narrowly, they are losing badly**
  (0.52–0.69 against 0.93), which no choice of reference operator explains away;
- and the cheapest thing anyone could do is competitive, which is worth knowing
  before spending 1.8 seconds a frame.

### More exposures make every method worse

| Frames | Unrecoverable | Mertens | Debevec + Reinhard | Debevec + Drago | Debevec + Mantiuk |
|---:|---:|---:|---:|---:|---:|
| 2 | 6.82% | **24.10** | **18.58** | **18.40** | **16.87** |
| 3 | 4.39% | 22.39 | 17.48 | 14.67 | 13.67 |
| 5 | **0.50%** | 20.92 | 14.70 | 11.51 | 9.89 |

Read the second column against the rest. **A wider bracket records more of the
scene** — the fraction lost in every frame falls from 6.8% to 0.5% — and **every
method gets worse anyway.** Mertens drops 3.2 dB, Mantiuk 7.0.

The extra frames are the ±4-stop ones, and at +4 stops **49.5% of the frame is
saturated**. None of these methods discounts a frame for being mostly clipped;
they weight per pixel, and a pixel that reads 255 looks confidently bright
rather than broken. So the information is genuinely there and the fusion makes
worse use of it.

*On two scenes Mertens reverses and improves with more frames. The effect is an
average over scenes, not a property of each — recorded because it is exactly the
kind of claim a small sample inverts.*

### What no method can recover

| Bracket spread | Frames | Blown everywhere | Crushed everywhere | **Unrecoverable** |
|---:|---:|---:|---:|---:|
| ±0 stops | 1 | 27.8% | 1.3% | **29.0%** |
| ±1 stop | 3 | 11.9% | 0.5% | **12.4%** |
| ±2 stops | 3 | 7.8% | 0.4% | **8.2%** |
| ±3 stops | 3 | 4.6% | 0.3% | **4.8%** |
| ±4 stops | 3 | 2.2% | 0.2% | **2.5%** |

A single exposure of a 12-stop scene loses **29% of it** — clipped to white or
buried under the noise floor, in the only frame there is. That is not an
algorithm's failure and no algorithm fixes it. It is the argument for bracketing,
stated as a number rather than as a picture.

### How the scenes were chosen

Twelve candidates grouped by where the range *sits* — one bright/dark boundary,
many small ones, subject against ground, an interior, or barely any range at
all — with the best of each family kept. All twelve cleared the 0.40 SSIM gate.

```
scene candidate lake and shrine · bright sky, dark foreground keep — best SSIM 0.943, 0.3% unrecoverable  [one boundary]
scene candidate rocky coast · sky over shadowed rock       keep — best SSIM 0.921, 0.3% unrecoverable  [one boundary]
scene candidate stone arch · many small bright/dark edges  keep — best SSIM 0.917, 0.8% unrecoverable  [many boundaries]
scene candidate windmills · white walls against sky        keep — best SSIM 0.930, 0.2% unrecoverable  [many boundaries]
scene candidate harbour and boat · sunlit town, shaded hull keep — best SSIM 0.931, 0.7% unrecoverable  [mixed]
scene candidate boat and shed · water reflections          keep — best SSIM 0.919, 0.8% unrecoverable  [mixed]
scene candidate temple dragon · lit statue, dark towers    keep — best SSIM 0.927, 0.7% unrecoverable  [subject vs ground]
scene candidate giraffe · lit animal, flat background      keep — best SSIM 0.931, 0.1% unrecoverable  [subject vs ground]
scene candidate gallery visitors · interior, lit pictures  keep — best SSIM 0.911, 0.6% unrecoverable  [interior]
scene candidate elephant in grass · even light, little range keep — best SSIM 0.924, 0.2% unrecoverable  [little range]
scene candidate squirrel on a rock · soft light, little range keep — best SSIM 0.925, 0.4% unrecoverable  [little range]
scene candidate woman and child · close faces, flat light  keep — best SSIM 0.932, 0.5% unrecoverable  [interior]
```

---

## What it does

```mermaid
flowchart LR
    A[Photograph] --> B[x illumination field<br/>12 stops of range]
    B --> C[TRUE radiance<br/>known exactly]
    C --> D[5 exposures<br/>each clips or crushes]
    C -.perfect information.-> E[Oracle: fixed tone curve]
    D --> F[4 fusion methods<br/>+ 2 controls]
    F --> G[Score vs the oracle]
    E --> G

    style C fill:#fef3c7,stroke:#d97706
    style E fill:#dbeafe,stroke:#2563eb
```

Because the scene is generated, three things are known that a real bracket
cannot give you:

| Known | Lets us ask |
|---|---|
| the exact radiance | how close is the output? (PSNR, SSIM vs the oracle) |
| which pixels clipped in every frame | **what was never recorded?** (the ceiling) |
| the exposure times | is the method using them correctly? |

## Limitations

* **The illumination field is synthetic and smooth.** A real high-range scene
  has hard boundaries — a window frame — not a sum of three sinusoids. Methods
  that handle soft gradients well may do relatively worse on a real bracket.
* **The oracle's tone curve is global**, and that biases the comparison toward
  global methods. Stated in the results rather than buried here.
* **Everything is perfectly aligned.** A handheld bracket is not, and alignment
  failure is the commonest way real HDR goes wrong. `cv2.createAlignMTB` exists
  and is not used, because with no misalignment to correct it would measure
  nothing.
* **No ghosting.** Nothing moves between frames. Ghost removal is a large part
  of practical HDR and is absent here for the same reason.

---

## Tests

20 tests, run with `pytest projects/11_hdr_exposure_fusion/tests -q`. They pin
the findings rather than the numbers: that the scene is genuinely wider than 8
bits, that no single exposure holds it, that the naive mean beats every real
method, that one exposure beats all three Debevec pipelines, and that a wider
bracket records more while every method makes worse use of it.

---

## Keywords

HDR · high dynamic range · exposure fusion · exposure bracketing · Mertens
fusion · Debevec Malik · camera response curve · radiance map · tone mapping ·
Reinhard · Drago · Mantiuk · clipping · classical computer vision · no deep
learning · OpenCV · Python · CPU only · reproducible image processing experiments

## References

* Mertens, Kautz & Van Reeth, *Exposure Fusion*, Pacific Graphics 2007.
* Debevec & Malik, *Recovering High Dynamic Range Radiance Maps from
  Photographs*, SIGGRAPH 1997.
* Reinhard, Stark, Shirley & Ferwerda, *Photographic Tone Reproduction for
  Digital Images*, SIGGRAPH 2002.
* Drago, Myszkowski, Annen & Chiba, *Adaptive Logarithmic Mapping for Displaying
  High Contrast Scenes*, Eurographics 2003.
* Mantiuk, Myszkowski & Seidel, *A Perceptual Framework for Contrast Processing
  of High Dynamic Range Images*, ACM TAP 2006.
