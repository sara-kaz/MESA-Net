#pragma once

#include <Arduino.h>

// Console abstraction — Seeed XIAO ESP32-S3
//
// The XIAO ESP32-S3 uses USB CDC (not USB-Serial/JTAG).
// With ARDUINO_USB_CDC_ON_BOOT=1 set in platformio.ini,
// Serial IS the USB CDC port and appears as /dev/cu.usbmodem*
// on macOS or COM* on Windows immediately after reset.
//
// Board appears as: "USB JTAG/serial debug unit" → NO
//                   "USB CDC"                    → YES (XIAO)
//
// No USBSerial / JTAG needed. Serial works reliably at 115200.

static inline void consoleBegin(uint32_t baud = 115200) {
    Serial.begin(baud);
    delay(100);  // Allow USB CDC enumeration on host
}

#define CONSOLE Serial

