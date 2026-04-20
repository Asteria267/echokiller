"""
EchoKiller ULTRA V2  —  Windows Edition
=======================================
Elite Acoustic Echo Cancellation Demo

Built to dominate Day 16 submissions.

FEATURES
✓ WAV input OR synthetic voice generation
✓ Premium dark UI dashboard
✓ Multi-path room echo simulation
✓ NLMS Adaptive FIR Filter
✓ Real DSP Metrics:
    - SNR Gain
    - ERLE
    - Runtime
✓ Better visual plots
✓ Audio playback
✓ Save output WAV
✓ Fast + stable + recruiter-grade

INSTALL
pip install numpy scipy matplotlib soundfile sounddevice

RUN
python echokiller_ultra_v2.py
python echokiller_ultra_v2.py --input voice.wav
python echokiller_ultra_v2.py --order 128
python echokiller_ultra_v2.py --save
"""

import argparse
import time
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================================
# OPTIONAL MODULES
# ==========================================================

try:
    import soundfile as sf
    HAS_SF = True
except:
    HAS_SF = False

try:
    import sounddevice as sd
    HAS_SD = True
except:
    HAS_SD = False

# ==========================================================
# CONFIG
# ==========================================================

SR = 16000
EPS = 1e-8
DEFAULT_ORDER = 128

# COLORS
BG = "#0b0f17"
PANEL = "#121826"
TXT = "#d8e1ff"
GRID = "#24304a"
CYAN = "#00e5ff"
PINK = "#ff4d8d"
GREEN = "#69ff47"
YELLOW = "#ffd54f"

# ==========================================================
# UI
# ==========================================================

def banner():
    print("\n" + "═"*72)
    print("   EchoKiller ULTRA V2   •   Adaptive Echo Canceller")
    print("   FIR + NLMS DSP Engine   •   BUILDCORED ORCAS")
    print("═"*72)

# ==========================================================
# HELPERS
# ==========================================================

def normalize(x):
    return (x / (np.max(np.abs(x)) + EPS) * 0.9).astype(np.float32)

def power_db(x):
    return 10*np.log10(np.mean(x**2) + EPS)

def snr(clean, test):
    noise = test - clean
    return 10*np.log10((np.mean(clean**2)+EPS)/(np.mean(noise**2)+EPS))

def erle(inp, out):
    return power_db(inp) - power_db(out)

# ==========================================================
# SIGNAL GENERATOR
# ==========================================================

def generate_voice(duration=5):
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)

    freqs = [120, 240, 360, 500, 800, 1200]
    amps  = [0.8, 0.45, 0.28, 0.2, 0.12, 0.08]

    y = np.zeros_like(t)

    for f, a in zip(freqs, amps):
        phase = np.random.rand() * 2*np.pi
        y += a*np.sin(2*np.pi*f*t + phase)

    env = 0.55 + 0.45*np.sin(2*np.pi*2*t)
    env *= 0.7 + 0.3*np.sin(2*np.pi*0.7*t + 1)

    y *= env

    noise = np.random.randn(len(t))*0.015
    y += signal.lfilter([1], [1, -0.97], noise)

    return normalize(y)

# ==========================================================
# ROOM ECHO MODEL
# ==========================================================

def add_echo(x):
    delays = [0.13, 0.26, 0.39, 0.51]
    gains  = [0.58, 0.32, 0.18, 0.09]

    out = x.copy()

    for d_sec, g in zip(delays, gains):
        d = int(d_sec * SR)
        temp = np.concatenate([np.zeros(d), x])[:len(x)]
        out += g * temp

    return normalize(out)

# ==========================================================
# NLMS FILTER
# ==========================================================

def nlms_filter(echo, desired, order):
    n = len(echo)

    w = np.zeros(order)
    buf = np.zeros(order)

    y = np.zeros(n)
    e = np.zeros(n)

    for i in range(n):
        buf[1:] = buf[:-1]
        buf[0] = echo[i]

        y[i] = np.dot(w, buf)
        e[i] = desired[i] - y[i]

        mu = 0.92 / (np.dot(buf, buf) + EPS)
        w += mu * e[i] * buf

    return normalize(y), w, e

# ==========================================================
# AUDIO
# ==========================================================

def load_audio(path):
    if not HAS_SF:
        print("Install soundfile first.")
        sys.exit()

    data, sr = sf.read(path)

    if len(data.shape) > 1:
        data = data[:,0]

    if sr != SR:
        data = signal.resample(data, int(len(data)*SR/sr))

    return normalize(data)

def play(x, label):
    if not HAS_SD:
        return
    print(f"\nPlaying {label}...")
    sd.play(x, SR)
    sd.wait()

# ==========================================================
# PLOTTING
# ==========================================================

def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TXT, fontsize=11)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4)

def dashboard(clean, echo, filt, coeffs, err, stats):
    fig = plt.figure(figsize=(16,9), facecolor=BG)
    fig.suptitle("EchoKiller ULTRA V2", color=GREEN, fontsize=20)

    axs = []

    for i in range(1,7):
        axs.append(plt.subplot(2,3,i))

    # waveforms
    style(axs[0], "Original")
    axs[0].plot(clean, color=CYAN)

    style(axs[1], "Echoed Input")
    axs[1].plot(echo, color=PINK)

    style(axs[2], "Filtered Output")
    axs[2].plot(filt, color=GREEN)

    # coeffs
    style(axs[3], "FIR Coefficients")
    axs[3].bar(range(len(coeffs)), coeffs, color=YELLOW)

    # error
    style(axs[4], "Convergence Error")
    axs[4].semilogy(err**2 + EPS, color=CYAN)

    # frequency response
    style(axs[5], "Frequency Response")
    w, h = signal.freqz(coeffs)
    axs[5].plot(w, 20*np.log10(np.abs(h)+EPS), color=GREEN)

    fig.text(0.5, 0.02, stats, ha="center", color=TXT, fontsize=11)

    plt.tight_layout(rect=[0,0.04,1,0.95])
    plt.show()

# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-play", action="store_true")

    args = parser.parse_args()

    banner()

    # source
    if args.input:
        print("\nLoading audio...")
        clean = load_audio(args.input)
    else:
        print("\nGenerating synthetic voice...")
        clean = generate_voice()

    print("Simulating room echo...")
    echo = add_echo(clean)

    print("Running NLMS adaptive FIR...")
    start = time.time()

    filt, coeffs, err = nlms_filter(echo, clean, args.order)

    runtime = time.time() - start

    # metrics
    before = snr(clean, echo)
    after = snr(clean, filt)
    gain = after - before
    cancel = erle(echo, filt)

    print("\nRESULTS")
    print("────────────────────────────")
    print(f"Taps             : {args.order}")
    print(f"Runtime          : {runtime:.3f}s")
    print(f"SNR Before       : {before:.2f} dB")
    print(f"SNR After        : {after:.2f} dB")
    print(f"SNR Gain         : +{gain:.2f} dB")
    print(f"Echo Reduction   : {cancel:.2f} dB")

    stats = (
        f"SNR Gain: +{gain:.2f} dB   |   "
        f"ERLE: {cancel:.2f} dB   |   "
        f"Runtime: {runtime:.3f}s   |   "
        f"Taps: {args.order}"
    )

    # playback
    if not args.no_play and HAS_SD:
        play(echo, "Echoed Audio")
        play(filt, "Filtered Audio")
        play(clean, "Original Audio")

    # save
    if args.save and HAS_SF:
        sf.write("echokiller_ultra_v2_output.wav", filt, SR)
        print("\nSaved: echokiller_ultra_v2_output.wav")

    dashboard(clean, echo, filt, coeffs, err, stats)

if __name__ == "__main__":
    main()