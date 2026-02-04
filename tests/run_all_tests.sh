#!/bin/bash
#
# Run all tests for Qwen3-ASR GGML
#
# Usage: ./tests/run_all_tests.sh [--verbose]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/models"
REF_DIR="$PROJECT_DIR/tests/reference"

# Default model paths
ASR_MODEL="$MODEL_DIR/qwen3-asr-0.6b-f16.gguf"
ALIGNER_MODEL="$MODEL_DIR/qwen3-forced-aligner-0.6b-f16.gguf"
SAMPLE_AUDIO="$PROJECT_DIR/sample.wav"

# Parse arguments
VERBOSE=0
if [[ "$1" == "--verbose" ]] || [[ "$1" == "-v" ]]; then
    VERBOSE=1
fi

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Print functions
print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_test() {
    echo -n "  Testing $1... "
}

print_pass() {
    echo -e "${GREEN}PASSED${NC}"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}FAILED${NC}"
    ((FAILED++))
}

print_skip() {
    echo -e "${YELLOW}SKIPPED${NC} ($1)"
    ((SKIPPED++))
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check build directory
    print_test "build directory"
    if [[ -d "$BUILD_DIR" ]]; then
        print_pass
    else
        print_fail
        echo "    Build directory not found. Run: mkdir build && cd build && cmake .. && make"
        exit 1
    fi
    
    # Check executables
    for exe in test_mel test_encoder test_decoder qwen3-asr-cli; do
        print_test "$exe executable"
        if [[ -x "$BUILD_DIR/$exe" ]]; then
            print_pass
        else
            print_fail
            echo "    Executable not found: $BUILD_DIR/$exe"
            exit 1
        fi
    done
    
    # Check models
    print_test "ASR model"
    if [[ -f "$ASR_MODEL" ]]; then
        print_pass
    else
        print_skip "model not found"
    fi
    
    print_test "ForcedAligner model"
    if [[ -f "$ALIGNER_MODEL" ]]; then
        print_pass
    else
        print_skip "model not found"
    fi
    
    # Check sample audio
    print_test "sample audio"
    if [[ -f "$SAMPLE_AUDIO" ]]; then
        print_pass
    else
        print_skip "sample.wav not found"
    fi
}

# Run mel spectrogram test
run_mel_test() {
    print_header "Mel Spectrogram Test"
    
    print_test "mel spectrogram computation"
    
    if [[ ! -f "$SAMPLE_AUDIO" ]]; then
        print_skip "sample.wav not found"
        return
    fi
    
    if [[ ! -f "$REF_DIR/mel.npy" ]] || [[ ! -f "$REF_DIR/mel_filters.npy" ]]; then
        print_skip "reference files not found"
        return
    fi
    
    cd "$PROJECT_DIR"
    if [[ $VERBOSE -eq 1 ]]; then
        if "$BUILD_DIR/test_mel" --audio "$SAMPLE_AUDIO" --reference "$REF_DIR/mel.npy" --filters "$REF_DIR/mel_filters.npy" --tolerance 1e-4; then
            print_pass
        else
            print_fail
        fi
    else
        if "$BUILD_DIR/test_mel" --audio "$SAMPLE_AUDIO" --reference "$REF_DIR/mel.npy" --filters "$REF_DIR/mel_filters.npy" --tolerance 1e-4 > /dev/null 2>&1; then
            print_pass
        else
            print_fail
        fi
    fi
}

# Run encoder test
run_encoder_test() {
    print_header "Audio Encoder Test"
    
    print_test "audio encoder forward pass"
    
    if [[ ! -f "$ASR_MODEL" ]]; then
        print_skip "ASR model not found"
        return
    fi
    
    if [[ ! -f "$REF_DIR/mel.npy" ]] || [[ ! -f "$REF_DIR/audio_features.npy" ]]; then
        print_skip "reference files not found"
        return
    fi
    
    cd "$PROJECT_DIR"
    if [[ $VERBOSE -eq 1 ]]; then
        if "$BUILD_DIR/test_encoder" --model "$ASR_MODEL" --mel "$REF_DIR/mel.npy" --ref "$REF_DIR/audio_features.npy" --tolerance 2e-2; then
            print_pass
        else
            print_fail
        fi
    else
        if "$BUILD_DIR/test_encoder" --model "$ASR_MODEL" --mel "$REF_DIR/mel.npy" --ref "$REF_DIR/audio_features.npy" --tolerance 2e-2 > /dev/null 2>&1; then
            print_pass
        else
            print_fail
        fi
    fi
}

# Run decoder test
run_decoder_test() {
    print_header "Text Decoder Test"
    
    print_test "text decoder forward pass"
    
    if [[ ! -f "$ASR_MODEL" ]]; then
        print_skip "ASR model not found"
        return
    fi
    
    cd "$PROJECT_DIR"
    if [[ $VERBOSE -eq 1 ]]; then
        if "$BUILD_DIR/test_decoder" --model "$ASR_MODEL"; then
            print_pass
        else
            print_fail
        fi
    else
        if "$BUILD_DIR/test_decoder" --model "$ASR_MODEL" > /dev/null 2>&1; then
            print_pass
        else
            print_fail
        fi
    fi
}

# Run CLI transcription test
run_cli_transcription_test() {
    print_header "CLI Transcription Test"
    
    print_test "CLI transcription"
    
    if [[ ! -f "$ASR_MODEL" ]]; then
        print_skip "ASR model not found"
        return
    fi
    
    if [[ ! -f "$SAMPLE_AUDIO" ]]; then
        print_skip "sample.wav not found"
        return
    fi
    
    cd "$PROJECT_DIR"
    OUTPUT=$(mktemp)
    
    if [[ $VERBOSE -eq 1 ]]; then
        if "$BUILD_DIR/qwen3-asr-cli" -m "$ASR_MODEL" -f "$SAMPLE_AUDIO" -o "$OUTPUT" --no-timing; then
            if [[ -s "$OUTPUT" ]]; then
                print_pass
                echo "    Output: $(head -c 100 "$OUTPUT")..."
            else
                print_fail
                echo "    Empty output"
            fi
        else
            print_fail
        fi
    else
        if "$BUILD_DIR/qwen3-asr-cli" -m "$ASR_MODEL" -f "$SAMPLE_AUDIO" -o "$OUTPUT" --no-timing 2>/dev/null; then
            if [[ -s "$OUTPUT" ]]; then
                print_pass
            else
                print_fail
            fi
        else
            print_fail
        fi
    fi
    
    rm -f "$OUTPUT"
}

# Run CLI alignment test
run_cli_alignment_test() {
    print_header "CLI Forced Alignment Test"
    
    print_test "CLI forced alignment"
    
    if [[ ! -f "$ALIGNER_MODEL" ]]; then
        print_skip "ForcedAligner model not found"
        return
    fi
    
    if [[ ! -f "$SAMPLE_AUDIO" ]]; then
        print_skip "sample.wav not found"
        return
    fi
    
    cd "$PROJECT_DIR"
    OUTPUT=$(mktemp)
    
    # Use a simple test text
    TEST_TEXT="Hello world"
    
    if [[ $VERBOSE -eq 1 ]]; then
        if "$BUILD_DIR/qwen3-asr-cli" -m "$ALIGNER_MODEL" -f "$SAMPLE_AUDIO" --align --text "$TEST_TEXT" -o "$OUTPUT" --no-timing; then
            if [[ -s "$OUTPUT" ]] && grep -q '"words"' "$OUTPUT"; then
                print_pass
                echo "    Output: $(head -c 200 "$OUTPUT")..."
            else
                print_fail
                echo "    Invalid JSON output"
            fi
        else
            print_fail
        fi
    else
        if "$BUILD_DIR/qwen3-asr-cli" -m "$ALIGNER_MODEL" -f "$SAMPLE_AUDIO" --align --text "$TEST_TEXT" -o "$OUTPUT" --no-timing 2>/dev/null; then
            if [[ -s "$OUTPUT" ]] && grep -q '"words"' "$OUTPUT"; then
                print_pass
            else
                print_fail
            fi
        else
            print_fail
        fi
    fi
    
    rm -f "$OUTPUT"
}

# Print summary
print_summary() {
    print_header "Test Summary"
    
    TOTAL=$((PASSED + FAILED + SKIPPED))
    
    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo "  --------"
    echo "  Total:   $TOTAL"
    echo ""
    
    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        return 1
    fi
}

# Main
main() {
    echo "Qwen3-ASR GGML Test Suite"
    echo "========================="
    
    check_prerequisites
    run_mel_test
    run_encoder_test
    run_decoder_test
    run_cli_transcription_test
    run_cli_alignment_test
    print_summary
}

main
