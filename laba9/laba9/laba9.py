from __future__ import annotations

import csv
import math
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.io import wavfile
from scipy import signal


# ============================================================
# Лабораторная работа №9.
# Анализ шума и шумопонижение звукового сигнала
# ============================================================

# Папка, куда можно положить свою запись музыкального инструмента.
# Лучше WAV, mono. Если файл stereo, код автоматически переведет его в mono.
INPUT_AUDIO_DIR = Path(__file__).resolve().parent / "input_audio"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results_lab9"
SRC_DIR = BASE_DIR / "src_lab9"
REPORT_PATH = BASE_DIR / "report_lab9.md"

AUDIO_DIR = RESULTS_DIR / "audio"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_DIR = RESULTS_DIR / "csv"

SRC_AUDIO_DIR = SRC_DIR / "audio"
SRC_PLOTS_DIR = SRC_DIR / "plots"
SRC_CSV_DIR = SRC_DIR / "csv"

# =========================
# ПАРАМЕТРЫ АНАЛИЗА
# =========================

# Если нет входного WAV, будет создан демонстрационный сигнал.
DEMO_SAMPLE_RATE = 44100
DEMO_DURATION_SEC = 5.0

# STFT:
# В лекции для спектрального вычитания рекомендуется окно Ханна около 50 мс
# и перекрытие около 75%.
WINDOW_MS = 50
OVERLAP = 0.75

# Коэффициент подавления шума k в спектральном вычитании.
NOISE_REDUCTION_K = 1.2

# Нижняя граница коэффициента подавления.
# Нужна, чтобы уменьшить эффект "музыкального шума".
GAIN_FLOOR = 0.06

# Сколько первых секунд использовать как шумовой фрагмент,
# если в начале записи есть тишина/фон.
NOISE_PROFILE_SECONDS = 0.5

# Параметры поиска максимумов энергии.
ENERGY_DT = 0.1
ENERGY_DF = 50.0
TOP_ENERGY_EVENTS = 20


@dataclass
class AudioInfo:
    input_file: str
    sample_rate: int
    channels_original: int
    duration_sec: float
    samples_count: int


@dataclass
class NoiseInfo:
    noise_rms: float
    signal_rms: float
    snr_before_db: float
    residual_noise_rms: float
    cleaned_rms: float
    snr_after_db: float


@dataclass
class EnergyEvent:
    rank: int
    time_start: float
    time_end: float
    freq_start: float
    freq_end: float
    energy: float


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def setup_dirs() -> None:
    for path in [
        RESULTS_DIR,
        SRC_DIR,
        AUDIO_DIR,
        PLOTS_DIR,
        CSV_DIR,
        SRC_AUDIO_DIR,
        SRC_PLOTS_DIR,
        SRC_CSV_DIR,
    ]:
        ensure_clean_dir(path)

    INPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def db_ratio(signal_power: float, noise_power: float) -> float:
    return 10.0 * math.log10((signal_power + 1e-12) / (noise_power + 1e-12))


def normalize_float_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        max_abs = max(abs(info.min), abs(info.max))
        y = x.astype(np.float64) / max_abs
    else:
        y = x.astype(np.float64)

    return np.clip(y, -1.0, 1.0)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    return np.round(x * 32767.0).astype(np.int16)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray, int]:
    sample_rate, data = wavfile.read(path)

    data_f = normalize_float_audio(data)

    if data_f.ndim == 1:
        channels = 1
        mono = data_f
    else:
        channels = data_f.shape[1]
        mono = data_f.mean(axis=1)

    mono = mono.astype(np.float64)

    # Убираем постоянную составляющую
    mono = mono - float(np.mean(mono))

    max_abs = float(np.max(np.abs(mono)) + 1e-12)
    mono = mono / max_abs * 0.95

    return sample_rate, mono, channels


def save_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    wavfile.write(path, sample_rate, float_to_int16(audio))


def find_input_audio() -> Path | None:
    supported = [".wav"]

    files = [
        p for p in sorted(INPUT_AUDIO_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in supported
    ]

    if not files:
        return None

    return files[0]


def generate_demo_audio() -> tuple[int, np.ndarray]:
    """
    Демонстрационный сигнал, если пользователь еще не положил свою запись.
    Имитирует музыкальный звук: основной тон + гармоники + шум.
    """

    sr = DEMO_SAMPLE_RATE
    t = np.linspace(0, DEMO_DURATION_SEC, int(sr * DEMO_DURATION_SEC), endpoint=False)

    # Несколько "нот"
    notes = [
        (0.3, 1.1, 440.0),
        (1.2, 2.0, 523.25),
        (2.1, 3.0, 659.25),
        (3.1, 4.5, 392.0),
    ]

    audio = np.zeros_like(t)

    for start, end, freq in notes:
        mask = (t >= start) & (t <= end)
        local_t = t[mask] - start

        duration = end - start
        attack = np.clip(local_t / 0.08, 0, 1)
        decay = np.exp(-local_t / max(0.3, duration))
        envelope = attack * decay

        tone = (
            1.0 * np.sin(2 * np.pi * freq * local_t)
            + 0.45 * np.sin(2 * np.pi * 2 * freq * local_t)
            + 0.22 * np.sin(2 * np.pi * 3 * freq * local_t)
            + 0.10 * np.sin(2 * np.pi * 4 * freq * local_t)
        )

        audio[mask] += envelope * tone

    rng = np.random.default_rng(6)

    stationary_noise = 0.035 * rng.normal(size=t.size)
    hum = 0.025 * np.sin(2 * np.pi * 50 * t)

    noisy_audio = audio + stationary_noise + hum

    noisy_audio = noisy_audio - float(noisy_audio.mean())
    noisy_audio = noisy_audio / (np.max(np.abs(noisy_audio)) + 1e-12) * 0.9

    return sr, noisy_audio.astype(np.float64)


def get_audio() -> tuple[AudioInfo, np.ndarray]:
    input_path = find_input_audio()

    if input_path is None:
        sample_rate, audio = generate_demo_audio()
        input_name = "demo_generated_instrument_with_noise.wav"
        input_file_path = AUDIO_DIR / input_name
        save_wav(input_file_path, sample_rate, audio)
        shutil.copy2(input_file_path, SRC_AUDIO_DIR / input_name)
        channels = 1
    else:
        sample_rate, audio, channels = load_wav_mono(input_path)
        input_name = input_path.name
        input_file_path = AUDIO_DIR / input_name
        save_wav(input_file_path, sample_rate, audio)
        shutil.copy2(input_file_path, SRC_AUDIO_DIR / input_name)

    info = AudioInfo(
        input_file=input_name,
        sample_rate=sample_rate,
        channels_original=channels,
        duration_sec=len(audio) / sample_rate,
        samples_count=len(audio),
    )

    return info, audio


def stft_params(sample_rate: int) -> tuple[int, int]:
    nperseg = int(round(sample_rate * WINDOW_MS / 1000.0))

    # Для удобства FFT делаем четное значение.
    if nperseg % 2 == 1:
        nperseg += 1

    noverlap = int(round(nperseg * OVERLAP))

    return nperseg, noverlap


def compute_stft(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg, noverlap = stft_params(sample_rate)

    f, t, zxx = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )

    return f, t, zxx


def inverse_stft(zxx: np.ndarray, sample_rate: int, original_length: int) -> np.ndarray:
    nperseg, noverlap = stft_params(sample_rate)

    _, reconstructed = signal.istft(
        zxx,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )

    if reconstructed.size < original_length:
        reconstructed = np.pad(reconstructed, (0, original_length - reconstructed.size))

    reconstructed = reconstructed[:original_length]
    reconstructed = reconstructed.astype(np.float64)

    max_abs = float(np.max(np.abs(reconstructed)) + 1e-12)
    if max_abs > 1.0:
        reconstructed = reconstructed / max_abs * 0.95

    return reconstructed


def spectral_subtraction(
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Шумопонижение методом спектрального вычитания.

    1. STFT с окном Ханна.
    2. Оценка спектра шума по начальному участку записи.
    3. Вычитание амплитудного спектра шума.
    4. Сохранение фазы исходного сигнала.
    5. Обратное STFT.
    """

    f, t, zxx = compute_stft(audio, sample_rate)

    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    noise_frames = np.where(t <= NOISE_PROFILE_SECONDS)[0]

    if noise_frames.size < 2:
        # Если запись слишком короткая, берем 10% кадров с минимальной энергией.
        frame_energy = np.mean(magnitude ** 2, axis=0)
        count = max(1, int(0.1 * frame_energy.size))
        noise_frames = np.argsort(frame_energy)[:count]

    noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)

    cleaned_magnitude = magnitude - NOISE_REDUCTION_K * noise_profile
    cleaned_magnitude = np.maximum(cleaned_magnitude, GAIN_FLOOR * magnitude)

    cleaned_zxx = cleaned_magnitude * np.exp(1j * phase)

    cleaned = inverse_stft(cleaned_zxx, sample_rate, len(audio))

    return cleaned, f, t, zxx, cleaned_zxx


def save_waveform_plot(audio: np.ndarray, sample_rate: int, title: str, path: Path) -> None:
    duration = len(audio) / sample_rate
    time = np.linspace(0.0, duration, len(audio), endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    ax.plot(time, audio, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда")

    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrum_plot(audio: np.ndarray, sample_rate: int, title: str, path: Path) -> None:
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)

    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    ax.plot(freqs, magnitude_db, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда, дБ")

    ax.set_xlim(20, min(sample_rate / 2, 20000))
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_spectrogram_plot(
    f: np.ndarray,
    t: np.ndarray,
    zxx: np.ndarray,
    title: str,
    path: Path,
) -> None:
    power_db = 20 * np.log10(np.abs(zxx) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    mesh = ax.pcolormesh(t, f, power_db, shading="auto")

    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")

    ax.set_yscale("log")
    ax.set_ylim(20, max(100, f[-1]))

    fig.colorbar(mesh, ax=ax, label="Амплитуда, дБ")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def estimate_noise_info(
    original: np.ndarray,
    cleaned: np.ndarray,
    sample_rate: int,
) -> NoiseInfo:
    noise_samples = max(1, int(NOISE_PROFILE_SECONDS * sample_rate))
    noise_fragment = original[:noise_samples]

    signal_fragment = original
    residual = original - cleaned

    noise_power = rms(noise_fragment) ** 2
    signal_power = rms(signal_fragment) ** 2

    residual_power = rms(residual) ** 2
    cleaned_power = rms(cleaned) ** 2

    snr_before = db_ratio(signal_power, noise_power)
    snr_after = db_ratio(cleaned_power, residual_power)

    return NoiseInfo(
        noise_rms=rms(noise_fragment),
        signal_rms=rms(signal_fragment),
        snr_before_db=snr_before,
        residual_noise_rms=rms(residual),
        cleaned_rms=rms(cleaned),
        snr_after_db=snr_after,
    )


def find_energy_events(
    f: np.ndarray,
    t: np.ndarray,
    zxx: np.ndarray,
    top_count: int = TOP_ENERGY_EVENTS,
) -> list[EnergyEvent]:
    """
    Поиск областей максимальной энергии с шагом:
    Δt = 0.1 с,
    Δf = 50 Гц.
    """

    power = np.abs(zxx) ** 2

    t_max = float(t[-1]) if t.size else 0.0
    f_max = float(f[-1]) if f.size else 0.0

    time_bins = np.arange(0.0, t_max + ENERGY_DT, ENERGY_DT)
    freq_bins = np.arange(0.0, f_max + ENERGY_DF, ENERGY_DF)

    candidates: list[tuple[float, float, float, float, float]] = []

    for ti in range(len(time_bins) - 1):
        t0 = time_bins[ti]
        t1 = time_bins[ti + 1]

        t_mask = (t >= t0) & (t < t1)

        if not np.any(t_mask):
            continue

        for fi in range(len(freq_bins) - 1):
            f0 = freq_bins[fi]
            f1 = freq_bins[fi + 1]

            # Пропускаем инфранизкие частоты, чтобы DC/гул не забивали таблицу.
            if f1 < 20:
                continue

            f_mask = (f >= f0) & (f < f1)

            if not np.any(f_mask):
                continue

            local_power = power[np.ix_(f_mask, t_mask)]

            energy = float(local_power.sum())

            candidates.append((energy, t0, t1, f0, f1))

    candidates.sort(key=lambda x: x[0], reverse=True)

    events: list[EnergyEvent] = []

    for rank, (energy, t0, t1, f0, f1) in enumerate(candidates[:top_count], start=1):
        events.append(
            EnergyEvent(
                rank=rank,
                time_start=t0,
                time_end=t1,
                freq_start=f0,
                freq_end=f1,
                energy=energy,
            )
        )

    return events


def save_energy_events_csv(events: list[EnergyEvent], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(
            [
                "rank",
                "time_start_sec",
                "time_end_sec",
                "freq_start_hz",
                "freq_end_hz",
                "energy",
            ]
        )

        for event in events:
            writer.writerow(
                [
                    event.rank,
                    f"{event.time_start:.3f}",
                    f"{event.time_end:.3f}",
                    f"{event.freq_start:.1f}",
                    f"{event.freq_end:.1f}",
                    f"{event.energy:.12f}",
                ]
            )


def save_summary_csv(audio_info: AudioInfo, noise_info: NoiseInfo, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(["parameter", "value"])

        writer.writerow(["variant", VARIANT])
        writer.writerow(["input_file", audio_info.input_file])
        writer.writerow(["sample_rate", audio_info.sample_rate])
        writer.writerow(["original_channels", audio_info.channels_original])
        writer.writerow(["duration_sec", f"{audio_info.duration_sec:.6f}"])
        writer.writerow(["samples_count", audio_info.samples_count])

        writer.writerow(["window_ms", WINDOW_MS])
        writer.writerow(["overlap", OVERLAP])
        writer.writerow(["noise_profile_seconds", NOISE_PROFILE_SECONDS])
        writer.writerow(["noise_reduction_k", NOISE_REDUCTION_K])
        writer.writerow(["gain_floor", GAIN_FLOOR])

        writer.writerow(["noise_rms", f"{noise_info.noise_rms:.10f}"])
        writer.writerow(["signal_rms", f"{noise_info.signal_rms:.10f}"])
        writer.writerow(["snr_before_db", f"{noise_info.snr_before_db:.6f}"])
        writer.writerow(["residual_noise_rms", f"{noise_info.residual_noise_rms:.10f}"])
        writer.writerow(["cleaned_rms", f"{noise_info.cleaned_rms:.10f}"])
        writer.writerow(["snr_after_db", f"{noise_info.snr_after_db:.6f}"])


def write_report(
    audio_info: AudioInfo,
    noise_info: NoiseInfo,
    events: list[EnergyEvent],
) -> None:
    lines: list[str] = []

    lines.append("# Лабораторная работа №9")
    lines.append("## Анализ шума")
    lines.append("")
    lines.append(f"### Вариант {VARIANT}")
    lines.append("")
    lines.append("### Исходные данные")
    lines.append("")
    lines.append(f"- Входной файл: `{audio_info.input_file}`")
    lines.append(f"- Частота дискретизации: `{audio_info.sample_rate}` Гц")
    lines.append(f"- Исходное число каналов: `{audio_info.channels_original}`")
    lines.append(f"- Длительность: `{audio_info.duration_sec:.3f}` с")
    lines.append(f"- Количество отсчетов: `{audio_info.samples_count}`")
    lines.append("")
    lines.append("Если в папке `input_audio` нет WAV-файла, программа создает демонстрационный зашумленный музыкальный сигнал.")
    lines.append("")
    lines.append("### Метод")
    lines.append("")
    lines.append("```text")
    lines.append("1. Сигнал переводится в mono.")
    lines.append("2. Строится STFT с окном Ханна.")
    lines.append("3. По начальному участку оценивается спектр шума.")
    lines.append("4. Выполняется спектральное вычитание:")
    lines.append("   Y[f,t] = max(X[f,t] - k * W[f], floor * X[f,t])")
    lines.append("5. Фаза берется от исходного сигнала.")
    lines.append("6. Сигнал восстанавливается через inverse STFT.")
    lines.append("```")
    lines.append("")
    lines.append("### Оценка шума")
    lines.append("")
    lines.append("| Параметр | Значение |")
    lines.append("|:--|--:|")
    lines.append(f"| RMS шума по начальному участку | {noise_info.noise_rms:.8f} |")
    lines.append(f"| RMS исходного сигнала | {noise_info.signal_rms:.8f} |")
    lines.append(f"| SNR до обработки, дБ | {noise_info.snr_before_db:.3f} |")
    lines.append(f"| RMS остатка после обработки | {noise_info.residual_noise_rms:.8f} |")
    lines.append(f"| RMS очищенного сигнала | {noise_info.cleaned_rms:.8f} |")
    lines.append(f"| SNR после обработки, дБ | {noise_info.snr_after_db:.3f} |")
    lines.append("")
    lines.append("CSV со сводкой: `results_lab9/csv/summary.csv`")
    lines.append("")
    lines.append("### 1. Осциллограммы")
    lines.append("")
    lines.append("| До обработки | После обработки |")
    lines.append("|:--:|:--:|")
    lines.append("| ![wave before](src_lab9/plots/waveform_before.png) | ![wave after](src_lab9/plots/waveform_after.png) |")
    lines.append("")
    lines.append("### 2. Спектры")
    lines.append("")
    lines.append("| До обработки | После обработки |")
    lines.append("|:--:|:--:|")
    lines.append("| ![spectrum before](src_lab9/plots/spectrum_before.png) | ![spectrum after](src_lab9/plots/spectrum_after.png) |")
    lines.append("")
    lines.append("### 3. Спектрограммы")
    lines.append("")
    lines.append("| До шумопонижения | После шумопонижения |")
    lines.append("|:--:|:--:|")
    lines.append("| ![spectrogram before](src_lab9/plots/spectrogram_before.png) | ![spectrogram after](src_lab9/plots/spectrogram_after.png) |")
    lines.append("")
    lines.append("### 4. Моменты максимальной энергии")
    lines.append("")
    lines.append(f"Поиск выполнен с шагом `Δt = {ENERGY_DT}` с и `Δf = {ENERGY_DF}` Гц.")
    lines.append("")
    lines.append("| Ранг | t0, c | t1, c | f0, Гц | f1, Гц | Энергия |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    for event in events[:10]:
        lines.append(
            f"| {event.rank} | "
            f"{event.time_start:.3f} | "
            f"{event.time_end:.3f} | "
            f"{event.freq_start:.1f} | "
            f"{event.freq_end:.1f} | "
            f"{event.energy:.6e} |"
        )

    lines.append("")
    lines.append("Полная таблица максимумов энергии: `results_lab9/csv/energy_events.csv`")
    lines.append("")
    lines.append("### 5. Восстановленная звуковая дорожка")
    lines.append("")
    lines.append("- Исходный mono WAV: `results_lab9/audio/original_mono.wav`")
    lines.append("- Очищенный WAV: `results_lab9/audio/cleaned_spectral_subtraction.wav`")
    lines.append("")
    lines.append("### Вывод")
    lines.append("")
    lines.append(
        "В ходе лабораторной работы был проанализирован зашумленный звуковой сигнал. "
        "Для него построены осциллограммы, спектры и спектрограммы до и после обработки. "
        "Шум был оценен по начальному участку записи, после чего применено спектральное "
        "вычитание с сохранением фазового спектра исходного сигнала. Восстановленная "
        "звуковая дорожка сохранена в WAV-файл. Также найдены временно-частотные области "
        "с наибольшей энергией при заданных шагах Δt и Δf."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_dirs()

    audio_info, audio = get_audio()

    cleaned, f_before, t_before, zxx_before, zxx_after = spectral_subtraction(
        audio,
        audio_info.sample_rate,
    )

    # Сохраняем исходный mono и очищенный звук
    original_mono_path = AUDIO_DIR / "original_mono.wav"
    cleaned_path = AUDIO_DIR / "cleaned_spectral_subtraction.wav"

    save_wav(original_mono_path, audio_info.sample_rate, audio)
    save_wav(cleaned_path, audio_info.sample_rate, cleaned)

    shutil.copy2(original_mono_path, SRC_AUDIO_DIR / "original_mono.wav")
    shutil.copy2(cleaned_path, SRC_AUDIO_DIR / "cleaned_spectral_subtraction.wav")

    # Графики
    save_waveform_plot(
        audio,
        audio_info.sample_rate,
        "Осциллограмма до обработки",
        PLOTS_DIR / "waveform_before.png",
    )

    save_waveform_plot(
        cleaned,
        audio_info.sample_rate,
        "Осциллограмма после шумопонижения",
        PLOTS_DIR / "waveform_after.png",
    )

    save_spectrum_plot(
        audio,
        audio_info.sample_rate,
        "Спектр до обработки",
        PLOTS_DIR / "spectrum_before.png",
    )

    save_spectrum_plot(
        cleaned,
        audio_info.sample_rate,
        "Спектр после шумопонижения",
        PLOTS_DIR / "spectrum_after.png",
    )

    save_spectrogram_plot(
        f_before,
        t_before,
        zxx_before,
        "Спектрограмма до шумопонижения",
        PLOTS_DIR / "spectrogram_before.png",
    )

    save_spectrogram_plot(
        f_before,
        t_before,
        zxx_after,
        "Спектрограмма после шумопонижения",
        PLOTS_DIR / "spectrogram_after.png",
    )

    for file_name in [
        "waveform_before.png",
        "waveform_after.png",
        "spectrum_before.png",
        "spectrum_after.png",
        "spectrogram_before.png",
        "spectrogram_after.png",
    ]:
        shutil.copy2(PLOTS_DIR / file_name, SRC_PLOTS_DIR / file_name)

    # Оценки шума и максимумов энергии
    noise_info = estimate_noise_info(audio, cleaned, audio_info.sample_rate)

    events = find_energy_events(
        f_before,
        t_before,
        zxx_before,
        top_count=TOP_ENERGY_EVENTS,
    )

    save_energy_events_csv(events, CSV_DIR / "energy_events.csv")
    save_summary_csv(audio_info, noise_info, CSV_DIR / "summary.csv")

    shutil.copy2(CSV_DIR / "energy_events.csv", SRC_CSV_DIR / "energy_events.csv")
    shutil.copy2(CSV_DIR / "summary.csv", SRC_CSV_DIR / "summary.csv")

    write_report(audio_info, noise_info, events)

    print("Лабораторная работа №9 выполнена.")
    print(f"Входной файл: {audio_info.input_file}")
    print(f"Частота дискретизации: {audio_info.sample_rate} Гц")
    print(f"Длительность: {audio_info.duration_sec:.3f} с")
    print(f"Исходных каналов: {audio_info.channels_original}")
    print(f"SNR до обработки: {noise_info.snr_before_db:.3f} дБ")
    print(f"SNR после обработки: {noise_info.snr_after_db:.3f} дБ")
    print(f"Очищенный звук: {cleaned_path}")
    print(f"Отчет: {REPORT_PATH}")
    print(f"Результаты: {RESULTS_DIR}")
    print("")
    print("Для своей записи положите WAV-файл в папку input_audio и запустите код снова.")


if __name__ == "__main__":
    main()