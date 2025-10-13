#!/bin/bash

# Directory containing ini files
INI_DIR="inis"

# Loop over all ini files and run CLASS
for ini_file in "$INI_DIR"/*.ini; do
    echo "Running CLASS with $ini_file ..."
    "class" "$ini_file" > "${ini_file%.ini}.log" 2>&1
done

echo "All runs completed."
