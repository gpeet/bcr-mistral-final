#!/bin/bash
set -euo pipefail

# Download OPIG antibody data files in parallel using wget.
#
# Usage:
#   ./01_download_opig.sh <url_list> <output_dir>
#
# Arguments:
#   url_list    - Text file with one download URL per line
#   output_dir  - Directory to save downloaded files

if [ $# -ne 2 ]; then
    echo "Usage: $0 <url_list> <output_dir>"
    exit 1
fi

url_list=$1
output_dir=$2

mkdir -p "$output_dir"

download_file() {
    local url=$1
    local output_dir=$2
    local file_name
    file_name=$(basename "$url")
    local output_path="$output_dir/$file_name"

    if [ -f "$output_path" ]; then
        echo "Already exists, skipping: $file_name"
    else
        wget -q -P "$output_dir" "$url"
    fi
}

export -f download_file

parallel --eta --jobs 16 download_file {} "$output_dir" < "$url_list"

echo "All downloads complete -> $output_dir"
