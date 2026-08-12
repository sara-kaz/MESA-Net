/**
 * Resource Guard Implementation
 *
 * Hardware-enforced resource gating for on-device training.
 */

#include "resource_guard.h"
#include "transport/console.h"

// ESP32-S3 temperature sensor API
// Arduino-ESP32 v2.x uses ESP-IDF 4.4 which has the legacy temp_sensor API.
// ESP-IDF v5.x+ uses the new temperature_sensor driver.
#if __has_include(<driver/temperature_sensor.h>)
#include <driver/temperature_sensor.h>
#include "transport/console.h"
#define TEMP_SENSOR_API_NEW 1
#elif __has_include(<driver/temp_sensor.h>)
#include <driver/temp_sensor.h>
#include "transport/console.h"
#define TEMP_SENSOR_API_LEGACY 1
#endif

// Global instance
ResourceGuard resourceGuard;

// ============================================================
// Constructor / Init
// ============================================================

ResourceGuard::ResourceGuard()
    : _last_train_ms(0),
      _total_episodes(0),
      _total_train_ms(0.0f) {
}

void ResourceGuard::begin() {
    _last_train_ms = 0;
    _total_episodes = 0;
    _total_train_ms = 0.0f;
    CONSOLE.println("[ResourceGuard] Resource guard initialized.");
    printStatus();
}

// ============================================================
// Temperature Reading
// ============================================================

float ResourceGuard::readTemperature() const {
    float tsens_value = 25.0f;  // Safe default

#if defined(TEMP_SENSOR_API_NEW)
    // ESP-IDF v5.x+ API
    temperature_sensor_handle_t temp_handle = NULL;
    temperature_sensor_config_t temp_config = TEMPERATURE_SENSOR_CONFIG_DEFAULT(-10, 80);
    esp_err_t err = temperature_sensor_install(&temp_config, &temp_handle);
    if (err == ESP_OK) {
        temperature_sensor_enable(temp_handle);
        temperature_sensor_get_celsius(temp_handle, &tsens_value);
        temperature_sensor_disable(temp_handle);
        temperature_sensor_uninstall(temp_handle);
    }
#elif defined(TEMP_SENSOR_API_LEGACY)
    // ESP-IDF v4.4 legacy API (Arduino-ESP32 v2.x)
    temp_sensor_config_t cfg = TSENS_CONFIG_DEFAULT();
    temp_sensor_set_config(cfg);
    temp_sensor_start();
    temp_sensor_read_celsius(&tsens_value);
    temp_sensor_stop();
#else
    // No temperature sensor API available — return safe default
    tsens_value = 25.0f;
#endif

    return tsens_value;
}

// ============================================================
// Resource Check
// ============================================================

ResourceCheckResult ResourceGuard::check(float last_inference_ms) const {
    ResourceCheckResult r = {};

    if (!AdaptiveConfig::ENABLE_RESOURCE_GATING) {
        // Resource gating disabled — always allow
        r.allowed = true;
        r.memory_ok = true;
        r.thermal_ok = true;
        r.timing_ok = true;
        r.cpu_ok = true;
        return r;
    }

    // 1. Memory check
#ifdef BOARD_HAS_PSRAM
    r.free_psram = ESP.getFreePsram();
#else
    r.free_psram = 0;
#endif
    r.free_heap = ESP.getFreeHeap();

    r.memory_ok = (r.free_psram >= AdaptiveConfig::MIN_FREE_PSRAM_BYTES) &&
                  (r.free_heap >= AdaptiveConfig::MIN_FREE_HEAP_BYTES);

    // 2. Thermal check
    r.temperature_c = readTemperature();
    r.thermal_ok = (r.temperature_c < AdaptiveConfig::MAX_TEMPERATURE_C);

    // 3. Timing check (cooldown)
    uint32_t now = millis();
    r.ms_since_last_train = (_last_train_ms > 0) ? (now - _last_train_ms) : UINT32_MAX;
    r.timing_ok = (r.ms_since_last_train >= AdaptiveConfig::MIN_TRAIN_INTERVAL_MS);

    // 4. CPU load check
    r.last_infer_ms = last_inference_ms;
    r.cpu_ok = (last_inference_ms <= AdaptiveConfig::MAX_INFERENCE_LATENCY_MS);

    // All must pass
    r.allowed = r.memory_ok && r.thermal_ok && r.timing_ok && r.cpu_ok;

    return r;
}

// ============================================================
// Training Lifecycle
// ============================================================

void ResourceGuard::markTrainEnd(float duration_ms) {
    _total_episodes++;
    _total_train_ms += duration_ms;
}

// ============================================================
// Status Print
// ============================================================

void ResourceGuard::printStatus() const {
    ResourceCheckResult r = check(0.0f);

    CONSOLE.println("[ResourceGuard] Current status:");
    CONSOLE.printf("  PSRAM free: %lu KB (min: %lu KB) %s\n",
                  (unsigned long)(r.free_psram / 1024),
                  (unsigned long)(AdaptiveConfig::MIN_FREE_PSRAM_BYTES / 1024),
                  r.memory_ok ? "OK" : "LOW");
    CONSOLE.printf("  Heap free:  %lu KB\n",
                  (unsigned long)(r.free_heap / 1024));
    CONSOLE.printf("  Temperature: %.1f C (max: %.1f C) %s\n",
                  r.temperature_c,
                  AdaptiveConfig::MAX_TEMPERATURE_C,
                  r.thermal_ok ? "OK" : "HOT");
    CONSOLE.printf("  Cooldown: %lu ms since last (min: %lu ms) %s\n",
                  (unsigned long)r.ms_since_last_train,
                  (unsigned long)AdaptiveConfig::MIN_TRAIN_INTERVAL_MS,
                  r.timing_ok ? "OK" : "WAIT");
    CONSOLE.printf("  Total episodes: %lu, avg duration: %.1f ms\n",
                  (unsigned long)_total_episodes,
                  avgTrainDuration());
}
