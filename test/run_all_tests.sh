#!/bin/bash
# Run all 4 NSFnets quick tests
# Usage: bash run_all_tests.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/opt/anaconda3/bin/python3

echo "=========================================="
echo "  NSFnets MindSpore - Full Test Suite"
echo "=========================================="
echo "Python: $($PYTHON --version)"
echo "Time:   $(date)"
echo ""

RESULTS=()
PASSED=0
FAILED=0

run_test() {
    local script=$1
    local name=$2
    echo ">>> Running: $name"
    echo "------------------------------------------"
    if $PYTHON "$SCRIPT_DIR/$script" 2>&1; then
        echo ""
        RESULTS+=("PASS: $name")
        ((PASSED++))
    else
        echo ""
        RESULTS+=("FAIL: $name")
        ((FAILED++))
    fi
    echo ""
}

# Run tests sequentially
run_test "test_case1_kovasznay.py"  "Case 1: Kovasznay Flow (2D Steady)"
run_test "test_case2_cylinder.py"   "Case 2: Cylinder Wake (2D Unsteady)"
run_test "test_case3_beltrami.py"   "Case 3: Beltrami Flow (3D Unsteady)"
run_test "test_case4_channel.py"    "Case 4: Turbulent Channel (3D Unsteady)"

# Summary
echo "=========================================="
echo "              TEST SUMMARY"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Total: $((PASSED + FAILED)) | Passed: $PASSED | Failed: $FAILED"
echo "=========================================="

[ $FAILED -eq 0 ] && exit 0 || exit 1
