# 🎧 EchoKiller ULTRA V2

> **Adaptive Acoustic Echo Cancellation System built in Python**
> A portfolio-grade Digital Signal Processing project that simulates and removes room echo using an **NLMS Adaptive FIR Filter**, inspired by the same core principles used in **speakerphones, smart speakers, video calls, hearing aids, and voice assistants**.

---

## 🚀 Project Overview

Echo is one of the most common problems in real-world audio systems.

When sound from speakers re-enters a microphone after a short delay, it creates repeated reflections and degraded speech quality. This is why phone calls sometimes sound hollow, doubled, or difficult to understand.

**EchoKiller ULTRA V2** demonstrates how this problem can be solved using adaptive signal processing.

The system:

* Generates or loads an audio signal
* Simulates room reflections / acoustic echo
* Learns the echo path using an **Adaptive FIR Filter**
* Removes the echo using **NLMS (Normalized Least Mean Squares)**
* Measures performance with industry-style metrics
* Visualizes results with professional plots
* Plays back before/after audio for comparison

This project recreates, in software, the type of logic embedded inside modern DSP hardware.

---

# 🧠 Why This Project Matters

Acoustic Echo Cancellation (AEC) is used in:

* 📞 Speakerphones
* 💻 Zoom / Teams / Google Meet calls
* 🎧 Headsets
* 🔊 Smart speakers
* 🦻 Hearing aids
* 📱 Smartphones
* 🎤 Voice-controlled devices

Without echo cancellation, many communication systems would be frustrating or unusable.

This project demonstrates the engineering concepts behind those products.

---

# ⚙️ Core Technologies Used

* **Python**
* **NumPy** – numerical processing
* **SciPy** – DSP utilities / signal tools
* **Matplotlib** – advanced visualization
* **SoundFile** – WAV audio reading/writing
* **SoundDevice** – real-time playback

---

# 🔬 DSP Concepts Demonstrated

## 1. FIR Filters (Finite Impulse Response)

The adaptive filter uses multiple coefficients (taps) to model delayed reflections.

It learns how past samples affect the current signal.

---

## 2. Adaptive Filtering

Unlike static filters, adaptive filters update themselves continuously.

They automatically learn the echo environment.

---

## 3. NLMS Algorithm

This project uses:

**Normalized Least Mean Squares**

An improved version of LMS that converges faster and remains stable under varying signal power.

Used heavily in real-time audio systems.

---

## 4. Echo Path Estimation

The system estimates how the room reflects sound:

* wall reflections
* delay paths
* attenuation
* multiple bounces

Then subtracts those learned reflections.

---

# 🏗️ How It Works (Pipeline)

## Step 1 — Source Signal

Either:

### A) Synthetic Voice-Like Audio

Generated mathematically using harmonics + envelopes.

OR

### B) Real WAV Audio Input

Your own voice/music/sample.

---

## Step 2 — Room Echo Simulation

The clean signal is copied through multiple delayed paths:

* first reflection
* second reflection
* weaker later reflections

This simulates a room or speakerphone environment.

---

## Step 3 — Adaptive Echo Cancellation

The NLMS filter learns the reflection pattern and reconstructs the unwanted echo.

Then it suppresses it.

---

## Step 4 — Evaluation

The program measures:

### SNR (Signal-to-Noise Ratio)

Higher = cleaner sound.

### ERLE

Echo Return Loss Enhancement

Measures how much echo was reduced.

### Runtime

Shows efficiency.

---

## Step 5 — Visualization Dashboard

Professional 2x3 panel dashboard:

1. Original waveform
2. Echoed waveform
3. Filtered waveform
4. FIR coefficients
5. Error convergence curve
6. Frequency response

---

# 📂 Project Structure

```text
EchoKiller_ULTRA_V2/
│── echokiller_ultra_v2.py
│── README.md
│── sample.wav (optional)
│── output.wav (generated if saved)
```

---

# 📦 Installation

## 1. Clone / Download

```bash
git clone <your-repository-url>
cd EchoKiller_ULTRA_V2
```

## 2. Install Dependencies

```bash
pip install numpy scipy matplotlib soundfile sounddevice
```

---

# ▶️ Usage

## Run with synthetic signal

```bash
python echokiller_ultra_v2.py
```

---

## Run with your own WAV file

```bash
python echokiller_ultra_v2.py --input voice.wav
```

---

## Increase filter size

```bash
python echokiller_ultra_v2.py --order 128
```

More taps = stronger modeling, slower runtime.

---

## Save cleaned output

```bash
python echokiller_ultra_v2.py --save
```

---

## Disable playback

```bash
python echokiller_ultra_v2.py --no-play
```

Useful for silent environments.

---

# 🖥️ Example Console Output

```text
EchoKiller ULTRA V2

Loading audio...
Simulating room echo...
Running NLMS adaptive FIR...

RESULTS
Taps             : 128
Runtime          : 0.311s
SNR Before       : 3.24 dB
SNR After        : 12.88 dB
SNR Gain         : +9.64 dB
Echo Reduction   : 11.72 dB
```

---

# 📊 Understanding the Graphs

## Original Signal

The clean reference audio.

---

## Echoed Input

Contains delayed reflections.

Often looks thicker / smeared.

---

## Filtered Output

Cleaner reconstructed signal after cancellation.

---

## FIR Coefficients

Represents the learned impulse response.

Spikes often correspond to echo delays.

---

## Convergence Error

Should trend downward.

Shows learning progress.

---

## Frequency Response

Displays how the filter behaves across frequencies.

---

# 🧪 Why NLMS Instead of Basic LMS?

Basic LMS uses fixed step sizes.

Problems:

* unstable on loud signals
* slower adaptation
* sensitive to scaling

NLMS normalizes by signal power:

### Benefits:

* faster convergence
* safer updates
* more robust performance
* better for real audio

---

# 💼 Resume Value

This project demonstrates:

* Digital Signal Processing
* Adaptive Algorithms
* Audio Engineering
* Data Visualization
* Scientific Python
* Real-world Systems Thinking
* Performance Metrics
* Problem Solving

---

## Example Resume Bullet

> Built an adaptive acoustic echo cancellation system in Python using NLMS FIR filtering, achieving measurable SNR improvement and real-time waveform diagnostics.

---

# 🎯 Why This Stands Out

Many beginner projects only:

* apply static filters
* plot one waveform
* use toy math

This project includes:

✅ adaptive learning
✅ metrics
✅ real DSP logic
✅ engineering visualization
✅ practical application

---

# 🔍 Possible Improvements (Future Versions)

* Real microphone live cancellation
* GUI desktop app
* Spectrogram visualization
* Double-talk detection
* Noise suppression stage
* Voice enhancement pipeline
* GPU acceleration
* Real-time streaming mode

---

# 🛠️ Troubleshooting

## soundfile missing

```bash
pip install soundfile
```

---

## playback not working

```bash
pip install sounddevice
```

---

## mp3 file unsupported

Convert to WAV first:

```bash
ffmpeg -i song.mp3 song.wav
```

---

## filter unstable

Try lower taps:

```bash
python echokiller_ultra_v2.py --order 64
```

---

# 📚 Educational Value

This project is excellent for learning:

* adaptive systems
* FIR filters
* convergence behavior
* acoustic modeling
* optimization
* real-world DSP applications

---

# 🧠 Inspiration

EchoKiller ULTRA V2 is inspired by technologies deployed in:

* Qualcomm DSP chips
* Smart speaker pipelines
* Teleconference systems
* Hearing enhancement devices

---

# 👨‍💻 Author

Developed as an advanced DSP portfolio project focused on combining:

**engineering depth + practical usefulness + clean presentation**

---

# ⭐ Final Thoughts

Echo cancellation is one of those invisible technologies most people use every day without noticing.

This project exposes the intelligence behind it.

By building EchoKiller ULTRA V2, you are not just filtering sound — you are recreating a core communications technology used across the modern world.

---

# ⭐ If You Like This Project

Consider adding:

* stars ⭐
* forks 🍴
* improvements 🚀
* experiments 🎧

---

# License

MIT License
