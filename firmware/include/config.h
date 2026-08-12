#pragma once

// Make this header self-contained for IntelliSense:
// Arduino builds typically pull these via Arduino.h, but the editor may parse this
// file standalone and otherwise report uint*_t as undefined.
#include <stdint.h>

// =============================
// Sampling / buffering settings
// =============================

// 100 Hz sampling => 10,000 us period
// This matches the model's expected input (1000 samples @ 100Hz = 10 seconds)
constexpr uint32_t SAMPLE_RATE_HZ = 100;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

// 10 seconds @ 100 Hz => 1000 samples
constexpr uint16_t WINDOW_SECONDS = 10;
constexpr uint16_t WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS;

// =============================
// Streaming settings
// =============================

// If true: prints a CSV line for every sample (100 Hz).
// Note: On ESP32-S3 "USB-Serial/JTAG" the sustained throughput can be limited
// (especially when the host monitor is set to 115200). Streaming 100 CSV lines/sec
// may cause stalls/resets and intermittent disconnect/reconnect in the serial monitor.
//
// For stable bring-up + inference verification, keep this false (only window markers
// + inference prints). You can enable it later once you increase throughput or switch
// to a faster/binary transport.
constexpr bool STREAM_EACH_SAMPLE = false;

// Enable binary protocol for faster streaming (future use)
constexpr bool USE_BINARY_PROTOCOL = false;

// Lightweight live telemetry stream (recommended for plotting)
// Prints one line every (1000 / LIVE_TELEMETRY_HZ) ms with key signals.
// Format (CSV):
//   LIVE,t_ms,ecg_adc,ppg_red,ppg_ir,gsr_adc,gsr_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,flags
constexpr bool ENABLE_LIVE_TELEMETRY = true;
constexpr uint16_t LIVE_TELEMETRY_HZ = 10;  // 10 Hz is stable over USB serial

// High-rate ECG telemetry (to avoid "squared" ECG plots).
// This is intentionally tiny so we can run it at 100 Hz without saturating 115200 baud.
// Format (CSV):
//   ECG,t_ms,ecg_adc_raw,ecg_adc,flags
constexpr bool ENABLE_ECG_TELEMETRY = true;
constexpr uint16_t ECG_TELEMETRY_HZ = SAMPLE_RATE_HZ;  // 100 Hz (matches sampling)

// High-rate PPG telemetry (so the pulse waveform is not "triangles").
// Keep this compact like ECG to fit 115200 baud.
// Format (CSV):
//   PPG,t_ms,ppg_red,ppg_ir,flags
constexpr bool ENABLE_PPG_TELEMETRY = true;
constexpr uint16_t PPG_TELEMETRY_HZ = SAMPLE_RATE_HZ;  // 100 Hz (matches MAX30102 config)

// Demo commands over Serial (for verifying the alert pipeline without "manufacturing"
// physiological conditions). Keys:
//   '1' or 's' -> STRESS alert
//   '3' or 'a' -> ARRHYTHMIA alert
//   '4' or 'c' -> CRITICAL alert
//   '0' or 'n' -> clear/no alert (prints a note)
constexpr bool ENABLE_DEMO_ALERT_COMMANDS = true;

// =============================
// Sensor Enable Flags
// =============================
// Use #define for preprocessor compatibility

// AD8232 ECG - Always enabled (primary sensor)
#define ENABLE_ECG 1

// MPU6050 IMU (accelerometer + gyroscope) - Always enabled
#define ENABLE_IMU 1

// MAX30102 Pulse Oximeter (I2C PPG) - Set to 1 when you add the sensor
// Provides Red and IR LED channels for heart rate and SpO2
#define ENABLE_MAX30102 1

// Analog PPG fallback - Use if you have an analog PPG sensor instead of MAX30102
// Set to 0 when using MAX30102
#define ENABLE_PPG_ANALOG 0

// GSR/EDA Sensor for stress detection - Set to 1 when you add the sensor
#define ENABLE_GSR 1

// =============================
// Sensor I2C Addresses
// =============================

// MPU6050 I2C address (0x68 if AD0=GND, 0x69 if AD0=VCC)
constexpr uint8_t MPU6050_I2C_ADDR = 0x68;

// MAX30102 I2C address (fixed at 0x57, cannot be changed)
constexpr uint8_t MAX30102_I2C_ADDR = 0x57;

// =============================
// On-Device Inference Settings
// =============================

// Enable on-device ML inference (requires model + sensors working)
#define ENABLE_INFERENCE 1

// =============================
// Profiling / benchmarking (for paper metrics)
// =============================
// When enabled, the firmware prints one CSV line per processed window with
// timing + memory metrics that can be parsed into latency/RAM/flash summaries.
//
// Lines start with:
//   PROF,<fields...>
//
// IMPORTANT:
// - For stable timing, consider enabling QUIET profiling mode below (reduces
//   serial printing during inference windows).
#define ENABLE_PROFILING 1

// If true: suppress verbose inference prints/boxes/alerts and print only the
// compact profiling CSV line per window.
constexpr bool PROFILING_QUIET_MODE = false;

// Print profiling line every N windows (1 = every window)
constexpr uint8_t PROFILING_WINDOW_INTERVAL = 1;

// Include per-sample acquisition timing stats (mean/max) in profiling output.
// This adds a few integer ops per sample, but is still low overhead at 100 Hz.
constexpr bool PROFILING_TRACK_SAMPLE_TIMING = true;

// Run inference every N windows (1 = every window, 2 = every other, etc.)
constexpr uint8_t INFERENCE_INTERVAL = 1;

// =============================
// Alert Thresholds
// =============================

// Arrhythmia detection threshold (0.0-1.0, higher = more confident)
constexpr float ARRHYTHMIA_ALERT_THRESHOLD = 0.7f;

// Stress level for alert (0=baseline, 1=stress, 2=amusement, 3=meditation)
// Alert if stress class probability > threshold
constexpr float STRESS_ALERT_THRESHOLD = 0.6f;

// =============================
// On-Device Adaptive Training
// =============================
// Enable the adaptive training subsystem (patent: feedback-driven
// on-device learning with resource-constrained partial updates).
// When enabled, the system can adapt classification head weights
// based on user corrections and confidence drift detection.
#define ENABLE_ADAPTIVE_TRAINING 1  // Enabled for V5 architecture (3-layer heads)

// =============================
// Debug Settings
// =============================

// Print debug messages to Serial
constexpr bool DEBUG_SERIAL = true;

// Print raw sensor values (useful for calibration)
constexpr bool DEBUG_RAW_SENSORS = false;

// =============================
// ECG signal cleanup (display + inference stability)
// =============================
// AD8232 ECG can look noisy due to motion, electrode contact, and USB power noise.
// These settings apply lightweight filtering and ADC oversampling.
constexpr bool ENABLE_ECG_FILTER = true;
constexpr uint8_t ECG_ADC_OVERSAMPLE = 8;  // 4-16 is typical; higher = smoother but more CPU
constexpr float ECG_HIGHPASS_HZ = 0.5f;    // baseline wander removal (typical ECG HPF)
constexpr float ECG_LOWPASS_HZ = 40.0f;    // typical diagnostic-ish LPF for 100 Hz sampling

// Notch filter to suppress mains hum (pick 50Hz or 60Hz for your region)
constexpr bool ENABLE_ECG_NOTCH = true;
constexpr float ECG_MAINS_HZ = 60.0f;

// Spike suppression: clamp implausible single-sample jumps (motion/electrode pop)
// NOTE: Keep this OFF until your wiring is correct; if the ADC is on the wrong pin
// (stuck at 0/4095), spike-clamp can make the plot look like hard steps.
constexpr bool ENABLE_ECG_SPIKE_CLAMP = false;
constexpr uint16_t ECG_SPIKE_DELTA_ADC = 600;  // max step change per sample at 100 Hz

