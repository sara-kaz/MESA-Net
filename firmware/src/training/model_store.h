#pragma once

/**
 * Model Store: Dual-Slot Weight Versioning with Flash Persistence
 *
 * Patent-relevant component: Implements a two-slot model versioning
 * system on resource-constrained edge hardware.
 *
 * Key innovations (patent claims):
 *   1. Dual-slot architecture: "stable" (proven) and "candidate" (adapted)
 *      weight sets co-exist in PSRAM with atomic promotion.
 *   2. Flash persistence via ESP32 NVS: adapted weights survive power
 *      cycles, enabling long-term personalization.
 *   3. Generation tracking: each adaptation cycle increments a version
 *      counter, creating an auditable adaptation history.
 *   4. Rollback mechanism: if candidate fails validation, the stable
 *      slot is restored in O(1) — no retraining needed.
 *   5. Factory reset: original pre-trained weights from flash ROM can
 *      always be restored, providing a guaranteed safe baseline.
 *
 * Memory layout:
 *   Stable slot:    PSRAM — the currently deployed, validated weights
 *   Candidate slot: PSRAM — weights being adapted (may not be validated)
 *   Flash backup:   NVS   — persisted stable weights across reboots
 *   Factory:        ROM   — original trained weights (read-only .inc files)
 */

#include <Arduino.h>
#include <nvs_flash.h>
#include <nvs.h>
#include "training_config.h"
#include "gradient_engine.h"

// ============================================================
// Adaptation Event Log Entry
// ============================================================

struct AdaptationEvent {
    uint32_t timestamp_ms;
    uint32_t generation;
    uint8_t  task_id;
    float    stable_accuracy;
    float    candidate_accuracy;
    bool     promoted;          // true = candidate promoted to stable
    bool     rolled_back;       // true = candidate discarded
};

// ============================================================
// Model Store Class
// ============================================================

class ModelStore {
public:
    ModelStore();
    ~ModelStore();

    /**
     * Initialize the model store.
     * - Initializes NVS flash partition
     * - Attempts to load previously adapted weights from NVS
     * - Falls back to factory weights if NVS is empty or corrupt
     * @return true on success
     */
    bool begin();

    /**
     * Snapshot current gradient engine weights into the candidate slot.
     * Call this after a training episode before validation.
     */
    bool snapshotCandidate();

    /**
     * Promote candidate weights to stable slot.
     * Called after successful validation.
     * Also persists to NVS flash.
     * @return true on success
     */
    bool promoteCandidate();

    /**
     * Rollback: discard candidate, restore stable weights to gradient engine.
     * Called when validation fails.
     */
    bool rollback();

    /**
     * Restore candidate weights from the candidate snapshot slot back
     * into the gradient engine. Used during A/B validation after a
     * temporary rollback to stable for evaluation.
     */
    bool restoreCandidate();

    /**
     * Factory reset: restore all weights from flash ROM (.inc files).
     * Clears NVS stored weights and resets generation counter.
     */
    bool factoryReset();

    /**
     * Persist current stable weights to NVS flash.
     * @return true on success
     */
    bool persistToFlash();

    /**
     * Load weights from NVS flash into the gradient engine.
     * @return true if valid weights were found and loaded
     */
    bool loadFromFlash();

    /**
     * Get current adaptation generation.
     */
    uint32_t generation() const { return _generation; }

    /**
     * Get last adaptation event.
     */
    const AdaptationEvent& lastEvent() const { return _last_event; }

    /**
     * Check if NVS has stored adapted weights.
     */
    bool hasPersistedWeights() const { return _has_persisted; }

    /**
     * Get total bytes used in the candidate/stable slots.
     */
    size_t slotSizeBytes() const { return _slot_size; }

    bool isReady() const { return _initialized; }

private:
    bool _initialized;
    bool _has_persisted;
    uint32_t _generation;
    size_t _slot_size;
    AdaptationEvent _last_event;

    // Candidate and stable snapshot buffers (PSRAM)
    float* _stable_snapshot;
    float* _candidate_snapshot;

    // NVS handle
    nvs_handle_t _nvs_handle;

    /** Calculate total weight bytes across all 3 task heads. */
    size_t calcTotalWeightBytes() const;

    /** Serialize all 3 task heads into a flat buffer. */
    bool serializeAllHeads(float* dest, size_t max_bytes);

    /** Deserialize flat buffer back into gradient engine. */
    bool deserializeAllHeads(const float* src, size_t num_bytes);
};

// Global instance
extern ModelStore modelStore;
