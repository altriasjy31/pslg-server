#!/bin/bash
ONTID=$1
ONTID=$ONTID<2?$ONTID:2

# Set environment variables
DATADIR=./data                    # Path to the dataset directory

if [ ! -e "$DATADIR" ]; then
  echo "Error: Address '$DATADIR' does not exist."
  exit 1
fi

DATE=7-26
ONTOLOGIES=(cc mf bp)                    # Ontologies: Cellular Component, Molecular Function, Biological Process
ONT=${ONTOLOGIES[$ONTID]}                     # Select ontology (e.g., bp)

# Configuration file for training
CONFIG=configs/training_msa-v1/${ONT}o-${DATE}.yml

# Training output directories
MODEL_DIR=$DATADIR/training/trained_model/msa-v1-${DATE}/${ONT}o/
MSA_DATA=$DATADIR/MSAs/             # Filtered MSAs directory
DATASET=$DATADIR/dataset_state_dict.pkl  # Dataset state dictionary

# Ensure output directories exist
if [ ! -e "$MODEL_DIR" ]; then
  mkdir -p $MODEL_DIR
fi

# Run training
echo "Starting training for ontology: ${ONT}..."
python scripts/construct_gendis.py -c $CONFIG \
    $DATASET \
    $MSA_DATA \
    $MODEL_DIR

echo "Training completed. Model saved to $MODEL_DIR"
