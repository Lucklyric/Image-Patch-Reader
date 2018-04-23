# Image-Patch-Reader
Read image patches in multi-threads in real time.

### Usage:
```python
test_reader = DataReaderPatchWise(
    path="/train/**/*.JPEG",  # path for your image folder 
    batch_size=64,
    patch_size=64,
    num_thread=2,
    num_image_per_sample=64,
    num_patches_per_image=10,
    min_cap_of_patches=1000,
    max_cap_of_patches=8000)

# Start all threads for feeding the patch queue
test_reader.start_feeding_q()

# Fecth on batch from the patche queue
sample_batch = test_reader.next_batch()

```

