IMAGE_SIZE=64
N_SAMPLES=2000000
CLEANUP=true

for DOWNSAMPLE_RATIO in 2 4 8 16 32; do
    echo "Processing downsample ratio: $DOWNSAMPLE_RATIO"
    papermill prep-dataset.ipynb prep-dataset-$DOWNSAMPLE_RATIO.ipynb \
     -p image_size $IMAGE_SIZE \
     -p downsample_ratio $DOWNSAMPLE_RATIO \
     -p n_samples $N_SAMPLES

    if [ "$CLEANUP" = true ]; then
        rm -f prep-dataset-$DOWNSAMPLE_RATIO.ipynb
    fi
done

echo "Combining datasets..."
python combine-datasets.py --image-size $IMAGE_SIZE \
    --ratios 2 4 8 16 32\
    --output ../data/combined_multi_res.npz