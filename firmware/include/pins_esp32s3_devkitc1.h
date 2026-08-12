#pragma once

// ============================================================
// ESP32-S3-DevKitC-1-N8R8 Pin Mapping (ADC-safe)
// ============================================================
//
// Hardware:
//   - ESP32-S3-DevKitC-1-N8R8
//   - AD8232 ECG (analog)
//   - MPU6050 6-axis IMU (I2C)
//   - MAX30102 Pulse Oximeter (I2C) [optional, for PPG]
//   - GSR/EDA Skin Conductance Sensor (analog) [optional, for stress]
//
// NOTE:
// On ESP32-S3, ADC-capable pins are GPIO 1-20.
// Many "classic ESP32" ADC pins (GPIO34/35/36/39) do NOT exist on S3.
//
// ADC1: GPIO 1-10 (recommended - no WiFi interference)
// ADC2: GPIO 11-20 (may conflict with WiFi)
//
// ============================================================

// --- ADC inputs (Analog sensors) ---

// AD8232 ECG output
constexpr int ECG_ADC_PIN = 1;  // ADC1_CH0 - Primary ECG signal

// GSR/EDA Sensor (for stress detection)
// Set to -1 if not using GSR sensor
constexpr int GSR_ADC_PIN = 2;  // ADC1_CH1 - Skin conductance (Grove GSR v1.2)

// Analog PPG fallback (if not using MAX30102)
// Set to -1 if using MAX30102 for PPG
constexpr int PPG_ADC_PIN = -1;  // Unused when MAX30102 is enabled

// If you ever use an *analog* 3-axis accelerometer, map here.
// For MPU6050 (I2C), these are unused.
constexpr int ACCEL_X_ADC_PIN = -1;
constexpr int ACCEL_Y_ADC_PIN = -1;
constexpr int ACCEL_Z_ADC_PIN = -1;

// --- AD8232 optional lead-off detection (digital inputs) ---
// LO+ and LO- are DIGITAL outputs from AD8232.
// HIGH = lead off (electrode not connected properly)
// Set to -1 to disable lead-off detection.
constexpr int AD8232_LO_PLUS_PIN = 4;   // GPIO 4 for LO+
constexpr int AD8232_LO_MINUS_PIN = 5;  // GPIO 5 for LO-

// --- I2C pins (shared bus for MPU6050 and MAX30102) ---
// Both MPU6050 (addr 0x68) and MAX30102 (addr 0x57) use I2C.
// IMPORTANT (ESP32-S3 DevKitC-1):
// Many boards do NOT expose GPIO22/23/24/25 (and some packages don't have them).
// Using a non-existent GPIO causes:
//   E (...) i2c: i2c_set_pin(...): scl gpio number error
//
// GPIO8/9 are commonly available and safe defaults on ESP32-S3 dev kits.
constexpr int I2C_SDA_PIN = 8;  // GPIO 8 - I2C Data
constexpr int I2C_SCL_PIN = 9;  // GPIO 9 - I2C Clock

// --- Optional interrupt pins ---
// MAX30102 interrupt (for FIFO ready notification)
// Set to -1 for polling mode
constexpr int MAX30102_INT_PIN = -1;

// MPU6050 interrupt (for motion detection)
// Set to -1 if not using interrupt mode
constexpr int MPU6050_INT_PIN = -1;

// --- Status LED (onboard or external) ---
constexpr int STATUS_LED_PIN = 48;  // ESP32-S3-DevKitC-1 has RGB LED on GPIO 48

// ============================================================
// Wiring Summary for ESP32-S3-DevKitC-1-N8R8:
// ============================================================
//
// AD8232 ECG Module:
//   VCC     -> 3.3V
//   GND     -> GND
//   OUTPUT  -> GPIO 1
//   LO+     -> GPIO 4 (optional, for lead-off detection)
//   LO-     -> GPIO 5 (optional, for lead-off detection)
//
// MPU6050 IMU (I2C):
//   VCC     -> 3.3V
//   GND     -> GND
//   SDA     -> GPIO 8
//   SCL     -> GPIO 9
//   AD0     -> GND (for address 0x68) or 3.3V (for 0x69)
//
// MAX30102 Pulse Oximeter (I2C) - Optional:
//   VIN     -> 3.3V
//   GND     -> GND
//   SDA     -> GPIO 8 (shared with MPU6050)
//   SCL     -> GPIO 9 (shared with MPU6050)
//   INT     -> Not connected (polling mode)
//
// GSR Sensor - Optional:
//   VCC     -> 3.3V
//   GND     -> GND
//   SIG/OUT -> GPIO 2
//
// ============================================================

