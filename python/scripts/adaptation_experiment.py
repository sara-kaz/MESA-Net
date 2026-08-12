#!/usr/bin/env python3
"""
On-Device Adaptive Training Experiment for IEEE JBHI Paper.

Collects before/after inference results across three phases:
  Phase 1: Baseline (no intervention) — N windows
  Phase 2: Corrections + Training — N windows with CORRECT + TRAIN commands
  Phase 3: Post-adaptation — N windows (observe adapted behavior)

Outputs a structured JSON + CSV for inclusion in the paper.

Usage:
  python3 scripts/adaptation_experiment.py --port /dev/cu.usbmodem1101
"""

import serial
import sys
import time
import json
import os
import argparse
from datetime import datetime

ACTIVITY_LABELS = ['Sedentary', 'Walking', 'Cycling', 'HighIntensity']
STRESS_LABELS = ['Baseline', 'Stress']
ARR_LABELS = ['Normal', 'Abnormal']

def parse_infer_line(line):
    """Parse an INFER CSV line into a dict.

    Format: INFER,t_ms,activity,act_conf,stress,str_conf,arrhythmia,arr_conf,
            is_moving,alert,str_logit0,str_logit1,arr_logit0,arr_logit1,calibrated
    Indices:  0     1      2       3       4       5        6          7
                    8      9       10       11       12       13        14
    """
    parts = line.split(',')
    if len(parts) < 15:
        return None
    try:
        return {
            'pred_activity': int(parts[2]),
            'activity_conf': float(parts[3]),
            'pred_stress': int(parts[4]),
            'stress_conf': float(parts[5]),
            'pred_arrhythmia': int(parts[6]),
            'arrhythmia_conf': float(parts[7]),
            'is_moving': int(parts[8]),
            'alert_type': int(parts[9]),
            'stress_logits': [float(parts[10]), float(parts[11])],
            'arrhythmia_logits': [float(parts[12]), float(parts[13])],
            'calibrated': int(parts[14]),
            'device_ms': int(parts[1]),
            'timestamp': time.time(),
        }
    except (ValueError, IndexError):
        return None


def wait_for_inference(ser, timeout=25):
    """Wait for the next INFER line and return parsed result.

    Reads in bulk to handle high-volume sensor data streams
    (200+ lines/sec of ECG/PPG data between INFER lines).
    """
    deadline = time.time() + timeout
    training_msgs = []
    partial = ""

    while time.time() < deadline:
        n = ser.in_waiting
        if n > 0:
            # Read all available bytes at once (up to 8KB)
            chunk = ser.read(min(n, 8192)).decode('utf-8', errors='replace')
            partial += chunk

            # Process complete lines
            while '\n' in partial:
                line, partial = partial.split('\n', 1)
                line = line.strip()

                if line.startswith('INFER,'):
                    result = parse_infer_line(line)
                    if result:
                        result['training_msgs'] = training_msgs
                        return result

                # Capture training-related messages
                if any(kw in line for kw in ['GradientEngine', 'Adaptive', 'TRAIN', 'CORRECT',
                                              'Rollback', 'Promoted', 'episode', 'loss', 'lockout']):
                    training_msgs.append(line)
                    print(f"    [TRAIN] {line}")
        else:
            time.sleep(0.005)

    return None


def send_command(ser, cmd):
    """Send a serial command and capture response."""
    ser.write(f"{cmd}\n".encode())
    time.sleep(0.15)
    responses = []
    partial = ""
    deadline = time.time() + 2.0
    while time.time() < deadline:
        n = ser.in_waiting
        if n > 0:
            chunk = ser.read(min(n, 8192)).decode('utf-8', errors='replace')
            partial += chunk
            while '\n' in partial:
                line, partial = partial.split('\n', 1)
                line = line.strip()
                if line:
                    responses.append(line)
                    if any(kw in line for kw in ['GradientEngine', 'Adaptive', 'TRAIN',
                                                  'Correction', 'episode', 'loss']):
                        print(f"    [RESP] {line}")
        else:
            time.sleep(0.02)
    return responses


def run_experiment(port, baud=115200, baseline_windows=5, adapt_windows=8, post_windows=5):
    """Run the three-phase adaptation experiment."""
    # Ground truth: sitting still with sensors = Sedentary, Baseline, Normal
    TRUE_ACT = 0   # Sedentary
    TRUE_STR = 0   # Baseline
    TRUE_ARR = 0   # Normal

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'training_results', 'adaptation_demo')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print("  ON-DEVICE ADAPTIVE TRAINING EXPERIMENT")
    print(f"  For IEEE JBHI Paper — {timestamp}")
    print(f"{'='*65}")
    print(f"\n  Ground truth: Sedentary / Baseline / Normal")
    print(f"  Phase 1: {baseline_windows} baseline windows (no intervention)")
    print(f"  Phase 2: {adapt_windows} windows with corrections + training")
    print(f"  Phase 3: {post_windows} post-adaptation windows")
    print(f"  Output: {output_dir}/\n")

    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)
    while ser.in_waiting:
        ser.readline()

    all_results = []
    phase_results = {1: [], 2: [], 3: []}

    # First send RESET to start from factory weights
    print("  Resetting to factory weights...")
    send_command(ser, "RESET")
    time.sleep(1)

    # ================================================================
    # PHASE 1: BASELINE
    # ================================================================
    print(f"\n{'='*65}")
    print("  PHASE 1: BASELINE (no intervention)")
    print(f"{'='*65}\n")

    for i in range(baseline_windows):
        print(f"  Waiting for window {i+1}/{baseline_windows}...")
        result = wait_for_inference(ser, timeout=30)
        if result is None:
            print("    TIMEOUT — no inference result")
            continue

        result['phase'] = 1
        result['window_num'] = i + 1
        result['corrections_sent'] = False

        act_correct = result['pred_activity'] == TRUE_ACT
        str_correct = result['pred_stress'] == TRUE_STR
        arr_correct = result['pred_arrhythmia'] == TRUE_ARR

        print(f"    Act={ACTIVITY_LABELS[result['pred_activity']]}({result['activity_conf']:.0%}) "
              f"{'OK' if act_correct else 'WRONG'} | "
              f"Str={STRESS_LABELS[result['pred_stress']]}({result['stress_conf']:.0%}) "
              f"{'OK' if str_correct else 'WRONG'} | "
              f"Arr={ARR_LABELS[result['pred_arrhythmia']]}({result['arrhythmia_conf']:.0%}) "
              f"{'OK' if arr_correct else 'WRONG'}")

        all_results.append(result)
        phase_results[1].append(result)

    # ================================================================
    # PHASE 2: CORRECTIONS + TRAINING
    # ================================================================
    print(f"\n{'='*65}")
    print("  PHASE 2: CORRECTIONS + TRAINING")
    print(f"{'='*65}\n")

    for i in range(adapt_windows):
        print(f"  Waiting for window {i+1}/{adapt_windows}...")
        result = wait_for_inference(ser, timeout=30)
        if result is None:
            print("    TIMEOUT — no inference result")
            continue

        result['phase'] = 2
        result['window_num'] = i + 1
        corrections = []

        # Send corrections for wrong predictions
        if result['pred_activity'] != TRUE_ACT:
            responses = send_command(ser, f"CORRECT 0 {TRUE_ACT}")
            corrections.append(f"activity->{ACTIVITY_LABELS[TRUE_ACT]}")

        if result['pred_stress'] != TRUE_STR:
            responses = send_command(ser, f"CORRECT 1 {TRUE_STR}")
            corrections.append(f"stress->{STRESS_LABELS[TRUE_STR]}")

        if result['pred_arrhythmia'] != TRUE_ARR:
            responses = send_command(ser, f"CORRECT 2 {TRUE_ARR}")
            corrections.append(f"arrhythmia->{ARR_LABELS[TRUE_ARR]}")

        result['corrections_sent'] = len(corrections) > 0
        result['corrections'] = corrections

        if corrections:
            print(f"    Corrections: {', '.join(corrections)}")
            # Trigger training
            print(f"    Sending TRAIN...")
            train_responses = send_command(ser, "TRAIN")
            result['train_responses'] = train_responses
            # Wait a moment for training to complete
            time.sleep(0.5)
            # Read any additional training output (bulk read)
            partial_buf = ""
            drain_deadline = time.time() + 1.5
            while time.time() < drain_deadline:
                n = ser.in_waiting
                if n > 0:
                    chunk = ser.read(min(n, 8192)).decode('utf-8', errors='replace')
                    partial_buf += chunk
                    while '\n' in partial_buf:
                        line, partial_buf = partial_buf.split('\n', 1)
                        line = line.strip()
                        if any(kw in line for kw in ['loss', 'episode', 'Promoted', 'Rollback', 'lockout']):
                            print(f"    [TRAIN] {line}")
                else:
                    break
        else:
            print(f"    All correct! No corrections needed.")

        act_correct = result['pred_activity'] == TRUE_ACT
        str_correct = result['pred_stress'] == TRUE_STR
        arr_correct = result['pred_arrhythmia'] == TRUE_ARR

        print(f"    Act={ACTIVITY_LABELS[result['pred_activity']]}({result['activity_conf']:.0%}) "
              f"{'OK' if act_correct else 'WRONG'} | "
              f"Str={STRESS_LABELS[result['pred_stress']]}({result['stress_conf']:.0%}) "
              f"{'OK' if str_correct else 'WRONG'} | "
              f"Arr={ARR_LABELS[result['pred_arrhythmia']]}({result['arrhythmia_conf']:.0%}) "
              f"{'OK' if arr_correct else 'WRONG'}")

        all_results.append(result)
        phase_results[2].append(result)

    # ================================================================
    # PHASE 3: POST-ADAPTATION
    # ================================================================
    print(f"\n{'='*65}")
    print("  PHASE 3: POST-ADAPTATION (no intervention)")
    print(f"{'='*65}\n")

    # Also check STATUS
    print("  Querying adaptation status...")
    send_command(ser, "STATUS")
    time.sleep(0.5)

    for i in range(post_windows):
        print(f"  Waiting for window {i+1}/{post_windows}...")
        result = wait_for_inference(ser, timeout=30)
        if result is None:
            print("    TIMEOUT — no inference result")
            continue

        result['phase'] = 3
        result['window_num'] = i + 1
        result['corrections_sent'] = False

        act_correct = result['pred_activity'] == TRUE_ACT
        str_correct = result['pred_stress'] == TRUE_STR
        arr_correct = result['pred_arrhythmia'] == TRUE_ARR

        print(f"    Act={ACTIVITY_LABELS[result['pred_activity']]}({result['activity_conf']:.0%}) "
              f"{'OK' if act_correct else 'WRONG'} | "
              f"Str={STRESS_LABELS[result['pred_stress']]}({result['stress_conf']:.0%}) "
              f"{'OK' if str_correct else 'WRONG'} | "
              f"Arr={ARR_LABELS[result['pred_arrhythmia']]}({result['arrhythmia_conf']:.0%}) "
              f"{'OK' if arr_correct else 'WRONG'}")

        all_results.append(result)
        phase_results[3].append(result)

    ser.close()

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}\n")

    for phase in [1, 2, 3]:
        results = phase_results[phase]
        if not results:
            continue
        n = len(results)
        act_acc = sum(1 for r in results if r['pred_activity'] == TRUE_ACT) / n * 100
        str_acc = sum(1 for r in results if r['pred_stress'] == TRUE_STR) / n * 100
        arr_acc = sum(1 for r in results if r['pred_arrhythmia'] == TRUE_ARR) / n * 100

        # Average confidences
        avg_str_conf = sum(r['stress_conf'] for r in results) / n
        avg_arr_conf = sum(r['arrhythmia_conf'] for r in results) / n

        phase_name = {1: "Baseline", 2: "Adaptation", 3: "Post-adapt"}[phase]
        print(f"  Phase {phase} ({phase_name}, {n} windows):")
        print(f"    Activity accuracy:   {act_acc:.0f}%")
        print(f"    Stress accuracy:     {str_acc:.0f}% (avg conf: {avg_str_conf:.3f})")
        print(f"    Arrhythmia accuracy: {arr_acc:.0f}% (avg conf: {avg_arr_conf:.3f})")
        print()

    # Save results
    output = {
        'experiment': 'on_device_adaptation',
        'timestamp': timestamp,
        'ground_truth': {'activity': TRUE_ACT, 'stress': TRUE_STR, 'arrhythmia': TRUE_ARR},
        'phases': {
            'baseline': {'n_windows': len(phase_results[1])},
            'adaptation': {'n_windows': len(phase_results[2])},
            'post_adaptation': {'n_windows': len(phase_results[3])},
        },
        'results': []
    }

    for r in all_results:
        entry = {
            'phase': r['phase'],
            'window': r['window_num'],
            'pred_activity': r['pred_activity'],
            'pred_stress': r['pred_stress'],
            'pred_arrhythmia': r['pred_arrhythmia'],
            'activity_conf': r['activity_conf'],
            'stress_conf': r['stress_conf'],
            'arrhythmia_conf': r['arrhythmia_conf'],
            'corrections_sent': r['corrections_sent'],
        }
        if 'corrections' in r:
            entry['corrections'] = r['corrections']
        if 'training_msgs' in r:
            entry['training_msgs'] = r['training_msgs']
        output['results'].append(entry)

    # Per-phase summary
    for phase in [1, 2, 3]:
        results = phase_results[phase]
        if not results:
            continue
        n = len(results)
        phase_key = {1: 'baseline', 2: 'adaptation', 3: 'post_adaptation'}[phase]
        output['phases'][phase_key]['activity_accuracy'] = sum(1 for r in results if r['pred_activity'] == TRUE_ACT) / n
        output['phases'][phase_key]['stress_accuracy'] = sum(1 for r in results if r['pred_stress'] == TRUE_STR) / n
        output['phases'][phase_key]['arrhythmia_accuracy'] = sum(1 for r in results if r['pred_arrhythmia'] == TRUE_ARR) / n
        output['phases'][phase_key]['avg_stress_conf'] = sum(r['stress_conf'] for r in results) / n
        output['phases'][phase_key]['avg_arrhythmia_conf'] = sum(r['arrhythmia_conf'] for r in results) / n

    json_path = os.path.join(output_dir, f'adaptation_experiment_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {json_path}")

    # Also save a CSV for easy plotting
    csv_path = os.path.join(output_dir, f'adaptation_experiment_{timestamp}.csv')
    with open(csv_path, 'w') as f:
        f.write('phase,window,pred_act,pred_str,pred_arr,act_conf,str_conf,arr_conf,corrections\n')
        for r in all_results:
            corr = '|'.join(r.get('corrections', []))
            f.write(f"{r['phase']},{r['window_num']},{r['pred_activity']},{r['pred_stress']},"
                    f"{r['pred_arrhythmia']},{r['activity_conf']:.4f},{r['stress_conf']:.4f},"
                    f"{r['arrhythmia_conf']:.4f},{corr}\n")
    print(f"  CSV:     {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='On-device adaptation experiment')
    parser.add_argument('--port', default='/dev/cu.usbmodem1101')
    parser.add_argument('--baseline', type=int, default=5, help='Baseline windows')
    parser.add_argument('--adapt', type=int, default=8, help='Adaptation windows')
    parser.add_argument('--post', type=int, default=5, help='Post-adaptation windows')
    args = parser.parse_args()

    run_experiment(args.port, baseline_windows=args.baseline,
                   adapt_windows=args.adapt, post_windows=args.post)
