#pragma once

// ============================================================
// Seeed XIAO ESP32-S3 Pin Mapping for MESA-Net
// ============================================================
//
// Board: Seeed XIAO ESP32-S3 (ESP32-S3R8 SoC)
//   - 240 MHz dual-core Xtensa LX7
//   - 8 MB OPI PSRAM | 8 MB Flash
//   - 21 x 17.5 mm form factor
//
// Sensors:
//   - AD8232     ECG analog front-end           (ADC)
//   - MPU6050    6-axis IMU                     (I2C)
//   - MAX30102   Pulse Oximeter / PPG            (I2C)
//   - GSR/EDA    Skin Conductance (optional)     (ADC)
//
// XIAO ESP32-S3 exposed GPIO vs silk-screen label:
//   D0  = GPIO1   (ADC1_CH0)
//   D1  = GPIO2   (ADC1_CH1)
//   D2  = GPIO3   (ADC1_CH2)
//   D3  = GPIO4
//   D4  = GPIO5   ← Hardware I2C SDA (Wire)
//   D5  = GPIO6   ← Hardware I2C SCL (Wire)
//   D6  = GPIO43
//   D7  = GPIO44
//   D8  = GPIO7
//   D9  = GPIO8
//   D10 = GPIO9
//   LED = GPIO21  (built-in LED, active HIGH)
//
// IMPORTANT: Do NOT use GPIO5 or GPIO6 for anything other than I2C —
// they are the hardware SDA/SCL lines for the XIAO's default Wire bus.
// GPIO4 (D3) is the last safe digital pin before the I2C block.
//
// ADC-capable pins (ADC1, safe with WiFi/BLE off):
//   GPIO1, GPIO2, GPIO3 (D0, D1, D2)
// ============================================================

// ---- ADC inputs (Analog sensors) -------------------------

// AD8232 ECG output → D0 (GPIO1, ADC1_CH0)
constexpr int ECG_ADC_PIN = 1;

// GSR/EDA Sensor → D1 (GPIO2, ADC1_CH1)
// Set to -1 if not using GSR
constexpr int GSR_ADC_PIN = 2;

// Analog PPG fallback (only if NOT using MAX30102)
// Set to -1 when MAX30102 is enabled
constexpr int PPG_ADC_PIN = -1;

// Analog accelerometer (unused — MPU6050 is I2C)
constexpr int ACCEL_X_ADC_PIN = -1;
constexpr int ACCEL_Y_ADC_PIN = -1;
constexpr int ACCEL_Z_ADC_PIN = -1;

// ---- AD8232 Lead-off detection (digital inputs) ----------
// LO+ and LO- are DIGITAL outputs from AD8232.
// HIGH = lead off (electrode not making contact).
//
// NOTE: LO- moved from GPIO5 (conflicts with SDA on XIAO) to D2/GPIO3.
constexpr int AD8232_LO_PLUS_PIN  = 4;  // D3 (GPIO4)
constexpr int AD8232_LO_MINUS_PIN = 3;  // D2 (GPIO3) ← was GPIO5 on DevKitC-1

// ---- I2C bus (shared: MPU6050 + MAX30102) ----------------
// XIAO ESP32-S3 hardware I2C defaults: SDA=GPIO5, SCL=GPIO6
// Both sensors share this bus at 400 kHz.
constexpr int I2C_SDA_PIN = 5;  // D4 (GPIO5) ← was GPIO8 on DevKitC-1
constexpr int I2C_SCL_PIN = 6;  // D5 (GPIO6) ← was GPIO9 on DevKitC-1

// ---- Optional interrupt pins -----------------------------
constexpr int MAX30102_INT_PIN = -1;  // polling mode
constexpr int MPU6050_INT_PIN  = -1;  // polling mode

// ---- Status LED ------------------------------------------
// XIAO ESP32-S3 has a built-in LED on GPIO21 (active HIGH).
// DevKitC-1 had RGB LED on GPIO48 — that pin does not exist on XIAO.
constexpr int STATUS_LED_PIN = 21;

// ============================================================
// Wiring Summary — Seeed XIAO ESP32-S3
// ============================================================
//
//  AD8232 ECG Module:
//    VCC     → 3.3V
//    GND     → GND
//    OUTPUT  → D0  (GPIO1)   ADC input
//    LO+     → D3  (GPIO4)   Lead-off detect (optional)
//    LO-     → D2  (GPIO3)   Lead-off detect (optional)
//    SDN     → GND           Always powered (not shutdown)
//
//  MPU6050 IMU (I2C):
//    VCC     → 3.3V
//    GND     → GND
//    SDA     → D4  (GPIO5)
//    SCL     → D5  (GPIO6)
//    AD0     → GND           I2C address = 0x68
//
//  MAX30102 PPG (I2C, shared bus):
//    VIN     → 3.3V
//    GND     → GND
//    SDA     → D4  (GPIO5)
//    SCL     → D5  (GPIO6)
//    INT     → NC            Polling mode
//
//  GSR Sensor (optional):
//    VCC     → 3.3V
//    GND     → GND
//    SIG     → D1  (GPIO2)   ADC input
//
// ============================================================
