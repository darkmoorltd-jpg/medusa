#!/bin/bash
# Convert raw datasets into MEDUSA standardised format
python -c "
from src.data.image_normaliser import normalise_xray, normalise_ct
import cv2, nibabel as nib, numpy as np
# Example: process all X-rays
print('Preprocessing complete (placeholder)')
"