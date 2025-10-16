#!/bin/bash

# Script to process HDF5 datasets for each environment
# Usage: bash process_datasets.sh

BASE_DIR="/mnt/nfs_client/minghuan/datasets/cpgen/datasets/generated"

# Function to convert environment name to abbreviated output name
get_output_name() {
    local env_name=$1
    local abbrev=""

    case $env_name in
        "ThreePieceAssemblyWide")
            abbrev="tpawide"
            ;;
        "ThreePieceAssembly_D1")
            abbrev="tpa-d1"
            ;;
        "SquareWide")
            abbrev="squarewide"
            ;;
        "Square_D1")
            abbrev="square-d1"
            ;;
        "SquareReal")
            abbrev="squarereal"
            ;;
        "StackThreeWide")
            abbrev="stackthreewide"
            ;;
        "StackThree_D1")
            abbrev="stackthree-d1"
            ;;
        "ThreadingWide")
            abbrev="threadingwide"
            ;;
        "Threading_D1")
            abbrev="threading-d1"
            ;;
        "KitchenWide")
            abbrev="kitchenwide"
            ;;
        "Kitchen_D1")
            abbrev="kitchen-d1"
            ;;
        "CoffeeWide")
            abbrev="coffeewide"
            ;;
        "Coffee_D1")
            abbrev="coffee-d1"
            ;;
        "MugCleanupWide")
            abbrev="mugcleanupwide"
            ;;
        "MugCleanup_D1")
            abbrev="mugcleanup-d1"
            ;;
        "HammerCleanupWide")
            abbrev="hammercleanupwide"
            ;;
        "HammerCleanup_D1")
            abbrev="hammercleanup-d1"
            ;;
        *)
            # Default: convert to lowercase and replace underscores with dashes
            abbrev=$(echo "$env_name" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
            ;;
    esac

    echo "$abbrev"
}

# Process each environment directory
for env_dir in "$BASE_DIR"/*; do
    if [ ! -d "$env_dir" ]; then
        continue
    fi

    env_name=$(basename "$env_dir")
    echo "========================================="
    echo "Processing environment: $env_name"
    echo "========================================="

    # Find RGB HDF5 files (exclude already processed files)
    rgb_files=$(find "$env_dir" -name "*-rgb-84-84.hdf5" -o -name "*-rgb-90-160.hdf5" 2>/dev/null)

    # Special handling for SquareReal which has depth-seg files
    if [ "$env_name" == "SquareReal" ]; then
        rgb_files=$(find "$env_dir" -name "depth-seg-*.hdf5" 2>/dev/null)
    fi

    # skip ThreePieceAssemblyWide
    if [ "$env_name" == "ThreePieceAssemblyWide" ]; then
        continue
    fi

    if [ -z "$rgb_files" ]; then
        echo "No RGB HDF5 files found in $env_name, skipping..."
        continue
    fi

    # Process each RGB file
    while IFS= read -r rgb_file; do
        if [ -z "$rgb_file" ]; then
            continue
        fi

        echo "Processing file: $rgb_file"

        # Get the directory containing the HDF5 file
        file_dir=$(dirname "$rgb_file")

        # Get abbreviated name for output
        abbrev=$(get_output_name "$env_name")

        if [ "$env_name" == "ThreePieceAssemblyWide" ]; then
            continue
        fi

        # Determine output name based on input file
        if [[ "$rgb_file" == *"rgb-84-84.hdf5" ]]; then
            output_name="${abbrev}-rgb-w-depth-ext-int.hdf5"
        elif [[ "$rgb_file" == *"rgb-90-160.hdf5" ]]; then
            output_name="${abbrev}-rgb-w-depth-ext-int.hdf5"
        elif [[ "$rgb_file" == *"depth-seg-90-160.hdf5" ]]; then
            output_name="${abbrev}-depth-seg-w-depth-ext-int.hdf5"
        else
            # Fallback: extract resolution from filename
            output_name="${abbrev}-rgb-w-depth-ext-int.hdf5"
        fi

        # Run the processing command
        echo "Running: python scripts/dataset_states_to_obs.py --dataset $rgb_file --output_name $output_name --depth --camera_names agentview robot0_eye_in_hand"

        python scripts/dataset_states_to_obs.py \
            --dataset "$rgb_file" \
            --output_name "$output_name" \
            --depth \
            --camera_names agentview robot0_eye_in_hand \
            --camera_width 256 \
            --camera_height 256

        if [ $? -eq 0 ]; then
            echo "Successfully processed: $rgb_file"
            echo "Output saved to: $file_dir/$output_name"
        else
            echo "ERROR: Failed to process $rgb_file"
        fi
        echo ""

    done <<< "$rgb_files"

done

echo "========================================="
echo "All environments processed!"
echo "========================================="
