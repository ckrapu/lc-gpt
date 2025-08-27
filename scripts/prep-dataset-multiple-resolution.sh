IMAGE_SIZE=128
N_SAMPLES_MAX=3000000
CLEANUP=true
N_GRID_UNIT=50

for DOWNSAMPLE_RATIO in 2 4 8 16 32; do
    echo "Processing downsample ratio: $DOWNSAMPLE_RATIO"
    papermill prep-dataset.ipynb prep-dataset-$DOWNSAMPLE_RATIO.ipynb \
     -p image_size $IMAGE_SIZE \
     -p downsample_ratio $DOWNSAMPLE_RATIO \
     -p n_sample_maxs $N_SAMPLES_MAX \
     -p n_grid_unit $N_GRID_UNIT \
     --log-output

    if [ "$CLEANUP" = true ]; then
        rm -f prep-dataset-$DOWNSAMPLE_RATIO.ipynb
    fi
done

echo "Combining datasets..."
python combine-datasets.py --image-size $IMAGE_SIZE \
    --ratios 2 4 8 16 32\
    --output ../data/combined_multi_res.npz