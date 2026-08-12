/**
 * Adaptive Trainer Implementation
 *
 * Orchestrates the complete on-device learning pipeline.
 */

#include "adaptive_trainer.h"
#include "transport/console.h"
#include <cstring>
#include <esp_heap_caps.h>

// Global instance
AdaptiveTrainer adaptiveTrainer;

// ============================================================
// Constructor
// ============================================================

AdaptiveTrainer::AdaptiveTrainer()
    : _initialized(false),
      _total_episodes(0),
      _current_lr(AdaptiveConfig::LEARNING_RATE) {
}

// ============================================================
// Initialization
// ============================================================

bool AdaptiveTrainer::begin() {
    if (_initialized) return true;

    CONSOLE.println();
    CONSOLE.println("╔═══════════════════════════════════════════╗");
    CONSOLE.println("║  On-Device Adaptive Training System       ║");
    CONSOLE.println("║  Patent: Feedback-Driven Edge Learning    ║");
    CONSOLE.println("╚═══════════════════════════════════════════╝");
    CONSOLE.println();

    // 1. Initialize gradient engine (must be first — others depend on it)
    if (!gradientEngine.begin()) {
        CONSOLE.println("[AdaptiveTrainer] FATAL: Gradient engine init failed");
        return false;
    }

    // 2. Initialize model store (needs gradient engine)
    if (!modelStore.begin()) {
        CONSOLE.println("[AdaptiveTrainer] FATAL: Model store init failed");
        return false;
    }

    // 3. Initialize feedback controller
    feedbackController.begin();

    // 4. Initialize resource guard
    resourceGuard.begin();

    // 5. Initialize validation engine
    validationEngine.begin();

    _current_lr = AdaptiveConfig::LEARNING_RATE;

    _initialized = true;

    CONSOLE.println();
    CONSOLE.println("[AdaptiveTrainer] All subsystems initialized.");
    CONSOLE.printf("[AdaptiveTrainer] Trainable params: %lu\n",
                  (unsigned long)gradientEngine.trainableParamCount());
    CONSOLE.printf("[AdaptiveTrainer] Learning rate: %.4f\n", _current_lr);
    CONSOLE.printf("[AdaptiveTrainer] Model generation: %lu\n",
                  (unsigned long)modelStore.generation());
    CONSOLE.println();
    CONSOLE.println("Serial commands:");
    CONSOLE.println("  CORRECT <task 0-2> <label>  — provide true label");
    CONSOLE.println("  TRAIN                        — force training episode");
    CONSOLE.println("  RESET                        — factory reset weights");
    CONSOLE.println("  ADAPT STATUS                 — print adaptation status");
    CONSOLE.println("  ADAPT UNLOCK                 — reset safety lockout");
    CONSOLE.println();

    return true;
}

// ============================================================
// Post-Inference Hook
// ============================================================

void AdaptiveTrainer::onInferenceComplete(const NNResult& result,
                                           const float* features,
                                           float infer_ms) {
    if (!_initialized) return;

    // 1. Cache features for potential user corrections
    feedbackController.cacheFeatures(features);

    // 2. Update drift tracking
    feedbackController.updateDrift(result);

    // 3. Check if training should be triggered
    TriggerReason reason;
    if (!feedbackController.shouldTrain(reason)) {
        return;  // No trigger — continue normal operation
    }

    // 4. Check if system is locked out
    if (validationEngine.isLockedOut()) {
        return;  // Safety lockout — skip
    }

    // 5. Check resource availability
    if (!resourceGuard.canTrain(infer_ms)) {
        if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
            auto r = resourceGuard.check(infer_ms);
            CONSOLE.printf("[AdaptiveTrainer] Training deferred: "
                          "mem=%s therm=%s time=%s cpu=%s\n",
                          r.memory_ok ? "ok" : "LOW",
                          r.thermal_ok ? "ok" : "HOT",
                          r.timing_ok ? "ok" : "WAIT",
                          r.cpu_ok ? "ok" : "BUSY");
        }
        return;  // Resources insufficient — defer
    }

    // 6. Run training episode
    EpisodeResult ep = runEpisode(reason);

    // 7. Log telemetry
    if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
        const char* reason_str = "unknown";
        switch (reason) {
            case TriggerReason::CONFIDENCE_DRIFT: reason_str = "drift"; break;
            case TriggerReason::USER_CORRECTION:  reason_str = "correction"; break;
            case TriggerReason::PERIODIC:          reason_str = "periodic"; break;
            case TriggerReason::MANUAL:            reason_str = "manual"; break;
            default: break;
        }

        CONSOLE.printf("%s,%lu,%lu,%s,%.4f,%.4f,%.1f,%s\n",
                      AdaptiveConfig::TELEMETRY_PREFIX,
                      (unsigned long)millis(),
                      (unsigned long)_total_episodes,
                      reason_str,
                      ep.avg_loss,
                      _current_lr,
                      ep.total_duration_ms,
                      ep.promoted ? "promoted" : (ep.rolled_back ? "rollback" : "validated"));
    }
}

// ============================================================
// Training Episode
// ============================================================

EpisodeResult AdaptiveTrainer::runEpisode(TriggerReason trigger) {
    EpisodeResult ep = {};
    ep.trigger = trigger;

    unsigned long t0 = micros();
    resourceGuard.markTrainStart();

    CONSOLE.println("[AdaptiveTrainer] === Training Episode Start ===");

    // Snapshot current weights as candidate starting point
    modelStore.snapshotCandidate();

    float total_loss = 0.0f;
    float total_grad_norm = 0.0f;
    uint8_t steps = 0;

    // Collect corrections into a PSRAM-allocated array for replay
    // (CorrectionSample is ~264 bytes each; too large for stack with MAX_CORRECTION_BUFFER=32)
    constexpr uint8_t MAX_LOCAL_CORRECTIONS = 8;  // Limit replay set
    CorrectionSample* corrections_local = (CorrectionSample*)heap_caps_malloc(
        MAX_LOCAL_CORRECTIONS * sizeof(CorrectionSample), MALLOC_CAP_SPIRAM);
    if (!corrections_local) {
        CONSOLE.println("[AdaptiveTrainer] Failed to allocate correction replay buffer");
        ep.steps_completed = 0;
        return ep;
    }
    uint8_t n_corrections = 0;
    {
        CorrectionSample tmp;
        while (feedbackController.popCorrection(tmp) &&
               n_corrections < MAX_LOCAL_CORRECTIONS) {
            corrections_local[n_corrections++] = tmp;
        }
    }

    if (n_corrections == 0) {
        CONSOLE.println("[AdaptiveTrainer] No corrections available");
    }

    // Train: replay corrections for up to STEPS_PER_EPISODE total steps
    uint8_t replay_idx = 0;
    while (n_corrections > 0 && steps < AdaptiveConfig::STEPS_PER_EPISODE) {
        // Check wall-clock budget
        float elapsed_ms = (micros() - t0) / 1000.0f;
        if (elapsed_ms > AdaptiveConfig::MAX_TRAIN_DURATION_MS) {
            CONSOLE.println("[AdaptiveTrainer] Training budget exceeded, stopping");
            break;
        }

        CorrectionSample& correction = corrections_local[replay_idx % n_corrections];
        replay_idx++;

        TrainingResult tr = gradientEngine.trainStep(
            correction.task_id,
            correction.features,
            correction.label,
            _current_lr
        );

        if (tr.success) {
            total_loss += tr.loss;
            total_grad_norm += tr.gradient_norm;
            steps++;

            // Add to validation buffer on first pass through corrections only
            if (replay_idx <= n_corrections) {
                validationEngine.addValidationSample(
                    correction.features, correction.task_id, correction.label);
            }
        } else {
            CONSOLE.printf("[AdaptiveTrainer] Step %d failed, skipping\n", steps);
            break;  // Don't retry failed steps
        }
    }

    ep.steps_completed = steps;
    ep.avg_loss = (steps > 0) ? total_loss / steps : 0.0f;
    ep.avg_gradient_norm = (steps > 0) ? total_grad_norm / steps : 0.0f;

    // Validate candidate
    if (steps > 0 && validationEngine.validationSampleCount() >= 2) {
        ValidationResult vr = validationEngine.validate();
        ep.validated = true;

        if (vr.promoted) {
            // Candidate passed validation — promote and persist
            modelStore.promoteCandidate();
            gradientEngine.incrementGeneration();
            ep.promoted = true;
        } else if (vr.weight_divergence) {
            // Weight divergence — hard rollback
            modelStore.rollback();
            ep.rolled_back = true;
            CONSOLE.println("[AdaptiveTrainer] Rolled back due to weight divergence");
        } else {
            // Failed accuracy check — rollback
            modelStore.rollback();
            ep.rolled_back = true;
        }
    } else if (steps > 0) {
        // Not enough validation data — tentatively accept
        // (will be validated after more corrections arrive)
        modelStore.snapshotCandidate();
        ep.validated = false;
        ep.promoted = false;

        if (AdaptiveConfig::LOG_TRAINING_EVENTS) {
            CONSOLE.println("[AdaptiveTrainer] Insufficient validation data, "
                           "tentatively accepting adaptation");
        }
    }

    // Decay learning rate
    _current_lr *= AdaptiveConfig::LR_DECAY_FACTOR;
    if (_current_lr < AdaptiveConfig::LEARNING_RATE_MIN) {
        _current_lr = AdaptiveConfig::LEARNING_RATE_MIN;
    }

    ep.total_duration_ms = (micros() - t0) / 1000.0f;

    resourceGuard.markTrainEnd(ep.total_duration_ms);
    feedbackController.markTrained();
    _total_episodes++;

    CONSOLE.printf("[AdaptiveTrainer] === Episode %lu Complete ===\n",
                  (unsigned long)_total_episodes);
    CONSOLE.printf("  Steps: %d, Avg loss: %.4f, Duration: %.1f ms\n",
                  ep.steps_completed, ep.avg_loss, ep.total_duration_ms);
    CONSOLE.printf("  Result: %s\n",
                  ep.promoted ? "PROMOTED" :
                  (ep.rolled_back ? "ROLLED BACK" : "pending validation"));

    // Free the replay buffer
    if (corrections_local) {
        heap_caps_free(corrections_local);
    }

    return ep;
}

EpisodeResult AdaptiveTrainer::forceTrainEpisode() {
    if (!_initialized) {
        CONSOLE.println("[AdaptiveTrainer] Not initialized");
        return {};
    }

    if (feedbackController.pendingCorrectionCount() == 0) {
        CONSOLE.println("[AdaptiveTrainer] No corrections in buffer — "
                       "nothing to train on");
        CONSOLE.println("  Use: CORRECT <task 0-2> <label> to provide labels");
        return {};
    }

    return runEpisode(TriggerReason::MANUAL);
}

// ============================================================
// Serial Command Handling
// ============================================================

bool AdaptiveTrainer::handleCommand(const char* command) {
    if (!command) return false;

    // Skip leading whitespace
    while (*command == ' ') command++;

    // CORRECT <task> <label>
    if (strncmp(command, "CORRECT", 7) == 0) {
        return parseCorrectCommand(command + 7);
    }

    // TRAIN
    if (strncmp(command, "TRAIN", 5) == 0) {
        forceTrainEpisode();
        return true;
    }

    // RESET
    if (strncmp(command, "RESET", 5) == 0) {
        CONSOLE.println("[AdaptiveTrainer] Factory reset requested...");
        modelStore.factoryReset();
        feedbackController.reset();
        validationEngine.reset();
        _current_lr = AdaptiveConfig::LEARNING_RATE;
        _total_episodes = 0;
        CONSOLE.println("[AdaptiveTrainer] Factory reset complete.");
        return true;
    }

    // ADAPT STATUS
    if (strncmp(command, "ADAPT STATUS", 12) == 0 ||
        strncmp(command, "STATUS", 6) == 0) {
        printStatus();
        return true;
    }

    // ADAPT UNLOCK
    if (strncmp(command, "ADAPT UNLOCK", 12) == 0 ||
        strncmp(command, "UNLOCK", 6) == 0) {
        validationEngine.resetLockout();
        CONSOLE.println("[AdaptiveTrainer] Safety lockout cleared.");
        return true;
    }

    return false;  // Command not recognized
}

bool AdaptiveTrainer::parseCorrectCommand(const char* args) {
    // Expected: " <task_id> <label>"
    int task_id = -1, label = -1;
    if (sscanf(args, " %d %d", &task_id, &label) != 2) {
        CONSOLE.println("Usage: CORRECT <task 0-2> <label>");
        CONSOLE.println("  Task 0 = Activity (label 0-7)");
        CONSOLE.println("  Task 1 = Stress   (label 0-3)");
        CONSOLE.println("  Task 2 = Arrhythmia (label 0-1)");
        return true;  // Handled (even if invalid)
    }

    if (task_id < 0 || task_id > 2) {
        CONSOLE.println("Invalid task_id (must be 0-2)");
        return true;
    }

    // Validate label range
    int max_label;
    switch (task_id) {
        case 0: max_label = NNConfig::ACTIVITY_CLASSES - 1; break;
        case 1: max_label = NNConfig::STRESS_CLASSES - 1; break;
        case 2: max_label = NNConfig::ARRHYTHMIA_CLASSES - 1; break;
        default: return true;
    }

    if (label < 0 || label > max_label) {
        CONSOLE.printf("Invalid label for task %d (must be 0-%d)\n",
                      task_id, max_label);
        return true;
    }

    if (!feedbackController.hasCachedFeatures()) {
        CONSOLE.println("No cached features — wait for an inference window first");
        return true;
    }

    // Store correction
    bool ok = feedbackController.addCorrection(
        task_id, label, feedbackController.cachedFeatures());

    if (ok) {
        // Also add to validation buffer
        validationEngine.addValidationSample(
            feedbackController.cachedFeatures(), task_id, label);

        const char* task_names[] = {"Activity", "Stress", "Arrhythmia"};
        CONSOLE.printf("[AdaptiveTrainer] Correction recorded: %s = %d\n",
                      task_names[task_id], label);
        CONSOLE.printf("  Pending corrections: %d\n",
                      feedbackController.pendingCorrectionCount());
    }

    return true;
}

// ============================================================
// Status Print
// ============================================================

void AdaptiveTrainer::printStatus() const {
    CONSOLE.println();
    CONSOLE.println("╔═══════════════════════════════════════════╗");
    CONSOLE.println("║  Adaptive Training Status                 ║");
    CONSOLE.println("╚═══════════════════════════════════════════╝");

    CONSOLE.printf("  Initialized:    %s\n", _initialized ? "YES" : "NO");
    CONSOLE.printf("  Adaptation:     %s\n",
                  validationEngine.isLockedOut() ? "LOCKED OUT" : "ENABLED");
    CONSOLE.printf("  Generation:     %lu\n",
                  (unsigned long)modelStore.generation());
    CONSOLE.printf("  Total episodes: %lu\n",
                  (unsigned long)_total_episodes);
    CONSOLE.printf("  Learning rate:  %.6f\n", _current_lr);
    CONSOLE.printf("  Trainable params: %lu\n",
                  (unsigned long)gradientEngine.trainableParamCount());
    CONSOLE.printf("  Persisted:      %s\n",
                  modelStore.hasPersistedWeights() ? "YES (NVS)" : "NO");

    CONSOLE.println("\n  Drift tracking:");
    const char* task_names[] = {"Activity", "Stress", "Arrhythmia"};
    for (int t = 0; t < 3; t++) {
        const auto& d = feedbackController.getDrift(t);
        CONSOLE.printf("    %s: EMA=%.3f %s (windows=%d)\n",
                      task_names[t],
                      d.confidence_ema,
                      d.drift_detected ? "DRIFT" : "ok",
                      d.window_count);
    }

    CONSOLE.printf("\n  Corrections pending: %d/%d\n",
                  feedbackController.pendingCorrectionCount(),
                  AdaptiveConfig::MAX_CORRECTION_BUFFER);
    CONSOLE.printf("  Validation samples:  %d/%d\n",
                  validationEngine.validationSampleCount(),
                  AdaptiveConfig::VALIDATION_BUFFER_SIZE);
    CONSOLE.printf("  Consecutive failures: %d/%d\n",
                  validationEngine.consecutiveFailures(),
                  AdaptiveConfig::MAX_CONSECUTIVE_FAILURES);

    CONSOLE.println("\n  Resource guard:");
    resourceGuard.printStatus();

    CONSOLE.println();
}
