#!/bin/bash

# check if urban100.zip is downloaded

missing_files=()

if [ ! -f data/compressed/urban100.zip ]; then
    echo "urban100.zip not found, download it from gdrive link listed in README.md"
    missing_files+=("urban100.zip")
fi

if [ ! -f data/compressed/BSDS200.zip ]; then
    echo "BSDS200.zip not found"
    missing_files+=("BSDS200.zip")
fi

if [ ! -f data/compressed/T91.zip ]; then
    echo "T91.zip not found"
    missing_files+=("T91.zip")
fi

if [ ! -f data/compressed/TEST.zip ]; then
    echo "TEST.zip not found"
    missing_files+=("TEST.zip")
fi

if [ ! -f data/compressed/TRAIN400.zip ]; then
    echo "TRAIN400.zip not found"
    missing_files+=("TRAIN400.zip")
fi

if [ ! -f data/compressed/CBSD68.zip ]; then
    echo "CBSD68.zip not found"
    missing_files+=("CBSD68.zip")
fi

if [ ! -f data/compressed/BSDS100.zip ]; then
    echo "BSDS100.zip not found"
    missing_files+=("BSDS100.zip")
fi  

if [ ! -f data/compressed/CBSD432.tar.gz ]; then
    echo "CBSD432.tar.gz not found"
    missing_files+=("CBSD432.tar.gz")
fi

if [ -n "$missing_files" ]; then
    echo "Missing files: ${missing_files[@]}"
    exit 1
fi

echo "All files are present"