/**
 * Model Store Implementation
 *
 * Dual-slot weight versioning with NVS flash persistence.
 */

#include "model_store.h"
#include "transport/console.h"

// Global instance
ModelStore modelStore;

// ============================================================
// Constructor / Destructor
// ============================================================

ModelStore::ModelStore()
    : _initialized(false),
      _has_persisted(false),
      _generation(0),
      _slot_size(0),
      _last_event({}),
      _stable_snapshot(nullptr),
      _candidate_snapshot(nullptr),
      _nvs_handle(0) {
}

ModelStore::~ModelStore() {
    if (_stable_snapshot) free(_stable_snapshot);
    if (_candidate_snapshot) free(_candidate_snapshot);
    if (_nvs_handle) nvs_close(_nvs_handle);
}

// ============================================================
// Helpers
// ============================================================

size_t ModelStore::calcTotalWeightBytes() const {
    if (!gradientEngine.isReady()) return 0;

    size_t total = 0;
    for (uint8_t t = 0; t < 3; t++) {
        const auto& h = gradientEngine.getTaskWeights(t);
        total += (h.input_dim * h.hidden1_dim + h.hidden1_dim +  // FC1
                  h.hidden1_dim + h.hidden1_dim +                // LN
                  h.hidden1_dim * h.hidden2_dim + h.hidden2_dim + // FC2
                  h.hidden2_dim * h.output_dim + h.output_dim    // FC3
                 ) * sizeof(float);
    }
    return total;
}

bool ModelStore::serializeAllHeads(float* dest, size_t max_bytes) {
    size_t offset = 0;
    for (uint8_t t = 0; t < 3; t++) {
        const auto& h = gradientEngine.getTaskWeights(t);
        size_t head_bytes = (h.input_dim * h.hidden1_dim + h.hidden1_dim +
                             h.hidden1_dim + h.hidden1_dim +
                             h.hidden1_dim * h.hidden2_dim + h.hidden2_dim +
                             h.hidden2_dim * h.output_dim + h.output_dim) * sizeof(float);
        if (offset + head_bytes > max_bytes) return false;

        if (!gradientEngine.snapshotWeights(t, dest + offset / sizeof(float),
                                             max_bytes - offset)) {
            return false;
        }
        offset += head_bytes;
    }
    return true;
}

bool ModelStore::deserializeAllHeads(const float* src, size_t num_bytes) {
    size_t offset = 0;
    for (uint8_t t = 0; t < 3; t++) {
        const auto& h = gradientEngine.getTaskWeights(t);
        size_t head_bytes = (h.input_dim * h.hidden1_dim + h.hidden1_dim +
                             h.hidden1_dim + h.hidden1_dim +
                             h.hidden1_dim * h.hidden2_dim + h.hidden2_dim +
                             h.hidden2_dim * h.output_dim + h.output_dim) * sizeof(float);
        if (offset + head_bytes > num_bytes) return false;

        if (!gradientEngine.restoreWeights(t, src + offset / sizeof(float),
                                            head_bytes)) {
            return false;
        }
        offset += head_bytes;
    }
    return true;
}

// ============================================================
// Initialization
// ============================================================

bool ModelStore::begin() {
    if (_initialized) return true;

    CONSOLE.println("[ModelStore] Initializing model versioning system...");

    if (!gradientEngine.isReady()) {
        CONSOLE.println("[ModelStore] ERROR: GradientEngine must be initialized first");
        return false;
    }

    // Calculate slot size
    _slot_size = calcTotalWeightBytes();
    CONSOLE.printf("[ModelStore] Weight slot size: %u bytes (%.1f KB)\n",
                  (unsigned)_slot_size, _slot_size / 1024.0f);

    // Allocate stable and candidate slots in PSRAM
    _stable_snapshot = (float*)ps_malloc(_slot_size);
    _candidate_snapshot = (float*)ps_malloc(_slot_size);

    if (!_stable_snapshot || !_candidate_snapshot) {
        CONSOLE.println("[ModelStore] ERROR: Failed to allocate snapshot slots");
        if (_stable_snapshot) { free(_stable_snapshot); _stable_snapshot = nullptr; }
        if (_candidate_snapshot) { free(_candidate_snapshot); _candidate_snapshot = nullptr; }
        return false;
    }

    // Initialize NVS
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        CONSOLE.println("[ModelStore] NVS needs erase, performing...");
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        CONSOLE.printf("[ModelStore] WARNING: NVS init failed (0x%x), persistence disabled\n", err);
        // Continue without persistence — in-memory adaptation still works
    }

    // Open NVS namespace
    err = nvs_open(AdaptiveConfig::NVS_NAMESPACE, NVS_READWRITE, &_nvs_handle);
    if (err != ESP_OK) {
        CONSOLE.printf("[ModelStore] WARNING: NVS open failed (0x%x)\n", err);
        _nvs_handle = 0;
    }

    // Try to load persisted weights
    if (_nvs_handle && loadFromFlash()) {
        CONSOLE.printf("[ModelStore] Loaded adapted weights from NVS (generation %lu)\n",
                      (unsigned long)_generation);
        _has_persisted = true;
    } else {
        CONSOLE.println("[ModelStore] No persisted weights found, using factory defaults");
        _has_persisted = false;
    }

    // Snapshot current (possibly loaded) weights as stable baseline
    serializeAllHeads(_stable_snapshot, _slot_size);

    _initialized = true;
    CONSOLE.println("[ModelStore] Model versioning system ready.");
    return true;
}

// ============================================================
// Candidate Management
// ============================================================

bool ModelStore::snapshotCandidate() {
    if (!_initialized) return false;
    return serializeAllHeads(_candidate_snapshot, _slot_size);
}

bool ModelStore::promoteCandidate() {
    if (!_initialized) return false;

    // Copy candidate -> stable
    memcpy(_stable_snapshot, _candidate_snapshot, _slot_size);

    // Increment generation
    _generation++;

    // Persist to flash
    bool persisted = persistToFlash();

    // Log event
    _last_event.timestamp_ms = millis();
    _last_event.generation = _generation;
    _last_event.promoted = true;
    _last_event.rolled_back = false;

    if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
        CONSOLE.printf("%s,%lu,%lu,promoted,%s\n",
                      AdaptiveConfig::ADAPT_PREFIX,
                      (unsigned long)millis(),
                      (unsigned long)_generation,
                      persisted ? "persisted" : "memory_only");
    }

    return true;
}

bool ModelStore::rollback() {
    if (!_initialized) return false;

    // Restore stable weights to gradient engine
    bool ok = deserializeAllHeads(_stable_snapshot, _slot_size);

    // Log event
    _last_event.timestamp_ms = millis();
    _last_event.generation = _generation;
    _last_event.promoted = false;
    _last_event.rolled_back = true;

    if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
        CONSOLE.printf("%s,%lu,%lu,rollback\n",
                      AdaptiveConfig::ADAPT_PREFIX,
                      (unsigned long)millis(),
                      (unsigned long)_generation);
    }

    return ok;
}

bool ModelStore::restoreCandidate() {
    if (!_initialized) return false;
    return deserializeAllHeads(_candidate_snapshot, _slot_size);
}

bool ModelStore::factoryReset() {
    if (!_initialized) return false;

    CONSOLE.println("[ModelStore] Factory reset: restoring original weights...");

    // Reset gradient engine to flash ROM weights
    gradientEngine.resetAllToOriginal();

    // Update stable snapshot
    serializeAllHeads(_stable_snapshot, _slot_size);

    // Clear NVS
    if (_nvs_handle) {
        nvs_erase_all(_nvs_handle);
        nvs_commit(_nvs_handle);
    }

    _generation = 0;
    _has_persisted = false;

    _last_event.timestamp_ms = millis();
    _last_event.generation = 0;
    _last_event.promoted = false;
    _last_event.rolled_back = false;

    if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
        CONSOLE.printf("%s,%lu,0,factory_reset\n",
                      AdaptiveConfig::ADAPT_PREFIX,
                      (unsigned long)millis());
    }

    CONSOLE.println("[ModelStore] Factory reset complete.");
    return true;
}

// ============================================================
// Flash Persistence (NVS)
// ============================================================

bool ModelStore::persistToFlash() {
    if (!_nvs_handle) return false;

    // Store generation counter
    esp_err_t err = nvs_set_u32(_nvs_handle, AdaptiveConfig::NVS_VERSION_KEY, _generation);
    if (err != ESP_OK) {
        CONSOLE.printf("[ModelStore] NVS write generation failed: 0x%x\n", err);
        return false;
    }

    // Store weight blob
    // NVS has a max blob size of ~508KB per entry (plenty for our ~26KB)
    err = nvs_set_blob(_nvs_handle, "weights", _stable_snapshot, _slot_size);
    if (err != ESP_OK) {
        CONSOLE.printf("[ModelStore] NVS write weights failed: 0x%x\n", err);
        return false;
    }

    err = nvs_commit(_nvs_handle);
    if (err != ESP_OK) {
        CONSOLE.printf("[ModelStore] NVS commit failed: 0x%x\n", err);
        return false;
    }

    _has_persisted = true;
    return true;
}

bool ModelStore::loadFromFlash() {
    if (!_nvs_handle) return false;

    // Read generation
    uint32_t gen = 0;
    esp_err_t err = nvs_get_u32(_nvs_handle, AdaptiveConfig::NVS_VERSION_KEY, &gen);
    if (err != ESP_OK) return false;

    // Read weight blob
    size_t required_size = _slot_size;
    err = nvs_get_blob(_nvs_handle, "weights", _stable_snapshot, &required_size);
    if (err != ESP_OK || required_size != _slot_size) {
        CONSOLE.printf("[ModelStore] NVS weight blob size mismatch or read error\n");
        return false;
    }

    // Restore into gradient engine
    if (!deserializeAllHeads(_stable_snapshot, _slot_size)) {
        CONSOLE.println("[ModelStore] Failed to deserialize NVS weights");
        return false;
    }

    _generation = gen;
    return true;
}
