#!/bin/bash

# Select ontology ID (pass as the first argument)
ONTID=$1
ONTID=$((ONTID<2 ? ONTID : 2))

# Set environment variables
DATADIR=./data                    # Path to the dataset directory
if [ ! -d "$DATADIR" ]; then
  echo "Error: Directory '$DATADIR' does not exist."
  echo "Please put the related data in '$DATADIR' or create a symbolic link."
  exit 1
fi

DATE=8-24
ONTOLOGIES=(cc mf bp)                    # Ontologies: Cellular Component, Molecular Function, Biological Process
ONT=${ONTOLOGIES[$ONTID]}                # Select ontology based on ID

# Configuration file for evaluation
CONFIG=configs/evaluating_msa-v1/${ONT}o-${DATE}.yml

# Evaluation directories and files
MODEL_DIR=$DATADIR/training/trained_model/msa-v1-${DATE}/${ONT}o/
MODEL_FILE=ema-fmax-highest.pth             # Trained model file
MSA_DATA=$DATADIR/MSAs/                  # Filtered MSAs directory
DATASET=$DATADIR/dataset_state_dict.pkl  # Dataset state dictionary
PRED_SAVE=$MODEL_DIR/evaluation_results.npy # Path to save average evaluation results

# Ensure the model directory exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model directory '$MODEL_DIR' does not exist."
  exit 1
fi

# Ensure the trained model file exists
if [ ! -e "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Error: Trained model '$MODEL_DIR/$MODEL_FILE' does not exist."
  exit 1
fi

# Run average evaluation
NUM_SAMPLINGS=5                          # Number of sampling passes for averaging
echo "Starting average evaluation for ontology: ${ONT}..."

python scripts/average_performance.py -c $CONFIG \
    -n $NUM_SAMPLINGS \
    --load $MODEL_FILE \
    -ps $PRED_SAVE \
    $DATASET \
    $MSA_DATA \
    $MODEL_DIR

if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully. Results saved to $PRED_SAVE."
else
  echo "Error occurred during evaluation."
fi
