# ESP32-S3 Multimodal Biomedical Monitoring Firmware

Firmware for **ESP32-S3-DevKitC-1-N8R8** to sample multiple biosensors:
- **ECG (AD8232)** via ADC - Heart electrical activity
- **IMU (MPU6050)** via I2C - 6-axis accelerometer + gyroscope for motion detection
- **PPG (MAX30102)** via I2C - Pulse oximetry (Red + IR LEDs) for heart rate and SpO2 [optional]
- **GSR/EDA Sensor** via ADC - Skin conductance for stress detection [optional]

The firmware buffers **10-second windows at 100Hz** (1000 samples) and streams samples over **Serial** (USB-C) in CSV format.

## Supported Hardware

| Sensor | Interface | Purpose | Status |
|--------|-----------|---------|--------|
| AD8232 | Analog (ADC) | ECG monitoring | Always enabled |
| MPU6050 | I2C | Motion/activity detection | Always enabled |
| MAX30102 | I2C | PPG/SpO2 | Optional (set `ENABLE_MAX30102=true`) |
| GSR/EDA | Analog (ADC) | Stress detection | Optional (set `ENABLE_GSR=true`) |

## ESP32-S3 ADC Pin Limitations

On **ESP32-S3**, only specific GPIOs support ADC:
- **ADC1**: GPIO **1–10** (recommended - no WiFi interference)
- **ADC2**: GPIO **11–20** (may conflict with WiFi)

Many "classic ESP32" ADC pins (e.g., GPIO34/35/36/39) **do not exist on ESP32-S3**.

This firmware uses ADC1 pins by default for reliability.

## Wiring Diagram

### AD8232 ECG Module → ESP32-S3

```
AD8232          ESP32-S3
-------         --------
VCC    ──────── 3.3V
GND    ──────── GND
OUTPUT ──────── GPIO 1  (ECG_ADC_PIN)
LO+    ──────── GPIO 4  (optional, lead-off detection)
LO-    ──────── GPIO 5  (optional, lead-off detection)
```

**Note:** LO+ and LO- are optional. They go HIGH when electrodes are not properly attached.

### MPU6050 IMU → ESP32-S3 (I2C)

```
MPU6050         ESP32-S3
-------         --------
VCC    ──────── 3.3V
GND    ──────── GND
SDA    ──────── GPIO 8  (I2C_SDA_PIN)
SCL    ──────── GPIO 9  (I2C_SCL_PIN)
AD0    ──────── GND     (for address 0x68)
```

### MAX30102 Pulse Oximeter → ESP32-S3 (I2C) [Optional]

```
MAX30102        ESP32-S3
--------        --------
VIN    ──────── 3.3V
GND    ──────── GND
SDA    ──────── GPIO 8  (shared I2C bus with MPU6050)
SCL    ──────── GPIO 9  (shared I2C bus with MPU6050)
INT    ──────── (not connected - polling mode)
```

**Note:** MAX30102 and MPU6050 share the same I2C bus. They have different I2C addresses (MAX30102=0x57, MPU6050=0x68) so there's no conflict.

### GSR/EDA Sensor → ESP32-S3 [Optional]

```
GSR Sensor      ESP32-S3
----------      --------
VCC    ──────── 3.3V
GND    ──────── GND
SIG/OUT ─────── GPIO 2  (GSR_ADC_PIN)
```

## Complete Wiring Summary

```
                    ESP32-S3-DevKitC-1-N8R8
                    ┌─────────────────────┐
                    │                     │
    AD8232 OUT ────►│ GPIO 1  (ADC1_CH0)  │
    AD8232 LO+ ────►│ GPIO 4              │ (optional)
    AD8232 LO- ────►│ GPIO 5              │ (optional)
                    │                     │
    GSR SIG    ────►│ GPIO 2  (ADC1_CH1)  │ (optional)
                    │                     │
    I2C SDA    ◄───►│ GPIO 8              │◄──── MPU6050 SDA
                    │                     │◄──── MAX30102 SDA
    I2C SCL    ◄───►│ GPIO 9              │◄──── MPU6050 SCL
                    │                     │◄──── MAX30102 SCL
                    │                     │
    3.3V       ◄────│ 3.3V                │───► All sensors VCC
    GND        ◄────│ GND                 │───► All sensors GND
                    │                     │
                    │ USB-C (Serial)      │ ← Data output
                    └─────────────────────┘
```

## Configuration

Edit `include/config.h` to enable/disable sensors:

```cpp
// Enable/disable sensors
constexpr bool ENABLE_ECG = true;        // AD8232 (always on)
constexpr bool ENABLE_IMU = true;        // MPU6050 (always on)
constexpr bool ENABLE_MAX30102 = false;  // Set to true when you add MAX30102
constexpr bool ENABLE_GSR = false;       // Set to true when you add GSR sensor
constexpr bool ENABLE_PPG_ANALOG = false; // Analog PPG fallback (if no MAX30102)
```

Edit `include/pins_esp32s3_devkitc1.h` if you need different pin assignments.

## Build / Flash (PlatformIO)

### Using VSCode
1. Install PlatformIO (VSCode extension)
2. Open `firmware/esp32` folder in VSCode
3. Click "Build" then "Upload"

### Using CLI

```bash
cd firmware/esp32
pio run                      # Build
pio run -t upload            # Flash
pio device monitor -b 115200 # Monitor serial output
```

## Serial Output Format

### CSV Header
```
t_us,ecg_adc,ppg_red,ppg_ir,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,gsr_adc,gsr_us,flags
```

### LIVE telemetry (recommended for real-time plotting)

For real-time visualization without overwhelming USB serial bandwidth, the firmware emits
low-rate telemetry lines (default **10 Hz**):

```
LIVE,t_ms,ecg_adc,ppg_red,ppg_ir,gsr_adc,gsr_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
```

You can plot these lines on your computer using:
`scripts/plots/live_serial_plot_ppg_gsr.py`

### Data Columns

| Column | Description | Units/Range |
|--------|-------------|-------------|
| t_us | Timestamp | Microseconds |
| ecg_adc | ECG raw ADC | 0-4095 (12-bit) |
| ppg_red | PPG Red LED | 0-262143 (18-bit MAX30102) or 0-4095 (analog) |
| ppg_ir | PPG IR LED | 0-262143 (MAX30102) or 0 (analog) |
| ax_g, ay_g, az_g | Acceleration | ±2g |
| gx_dps, gy_dps, gz_dps | Gyroscope | ±250 deg/sec |
| gsr_adc | GSR raw ADC | 0-4095 (12-bit) |
| gsr_us | GSR conductance | microSiemens (µS) |
| flags | Status flags | See below |

### Status Flags (bitmask)
- Bit 0 (0x01): ECG lead-off detected
- Bit 1 (0x02): PPG data valid
- Bit 2 (0x04): IMU data valid
- Bit 3 (0x08): GSR electrodes attached

### Window Markers

Every 10-second window (1000 samples):
```
WINDOW_START,0
<1000 CSV lines>
WINDOW_END,0
WINDOW_START,1
<1000 CSV lines>
WINDOW_END,1
...
```

## Sensor Status at Startup

The firmware prints sensor status on boot:
```
====================================
ESP32-S3 Multimodal Biomedical Monitor
====================================
AD8232 ECG: Initialized
MPU6050 IMU: Initialized
MAX30102 PPG: NOT DETECTED - check wiring!
GSR Sensor: Electrodes not attached
=== SENSOR STATUS ===
ECG (AD8232):   OK
IMU (MPU6050):  OK
PPG (MAX30102): NOT DETECTED
GSR:            NOT DETECTED
=====================
```

## Troubleshooting

### MPU6050 not detected
- Check VCC is 3.3V (not 5V)
- Verify SDA/SCL connections
- Check AD0 pin (GND for 0x68, VCC for 0x69)
- Try running I2C scanner sketch

### MAX30102 not detected
- Verify I2C connections (shares bus with MPU6050)
- Check part ID should be 0x15
- Ensure `ENABLE_MAX30102=true` in config.h

### ECG signal noisy
- Check electrode connections
- Use proper ECG electrodes (not just wires)
- Ensure good skin contact
- Check LO+/LO- for lead-off indication
- If you previously tied LO+ to 3.3V and LO- to GND: undo that. LO pins are **outputs** and should go to ESP32 GPIO inputs.
- USB power can inject noise; for cleaner ECG try powering from a battery bank/LiPo or ensure the subject is not touching grounded equipment.

### GSR reading stuck at 0
- Check electrode contact
- Verify reference resistor value in gsr_sensor.h
- Ensure skin is slightly moist (dry skin = very low conductance)

## Memory Usage (ESP32-S3-N8R8)

- Flash: ~300KB (plenty of room for model)
- RAM: ~50KB for buffers and variables
- PSRAM: 8MB available (if needed for larger models)

## Next Steps

1. **Add MAX30102**: Set `ENABLE_MAX30102=true`, wire sensor
2. **Add GSR**: Set `ENABLE_GSR=true`, wire sensor
3. **Deploy Model**: Convert trained model using `src/deployment/esp32_converter.py`
4. **Enable Inference**: Set `ENABLE_INFERENCE=true` after model deployment

## Demo alerts (serial commands)

To verify the alert-printing pipeline without forcing physiological conditions, type one of
these keys into the serial monitor:

- `1` or `s`: force **STRESS** alert
- `3` or `a`: force **ARRHYTHMIA** alert
- `4` or `c`: force **CRITICAL** alert
- `0` or `n`: clear/no alert
