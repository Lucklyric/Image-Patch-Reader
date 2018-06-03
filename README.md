# Image-Patch-Reader
Read image patches in multi-threads in real time.

### Usage Multi-Thread Version:
```python
from data_reader import DataReaderPatchWise
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

# Fecth one batch from the patche queue
sample_batch = test_reader.next_batch()

# Stop all threads 
test_reader.stop_feeding_q()

```
### Usage Multi-Process Version:
```python
from data_reader_mp import DataReaderPatchWiseMP
test_reader = DataReaderPatchWiseMP(
    path="/train/**/*.JPEG",  # path for your image folder
    batch_size=64,
    patch_size=64,
    num_process=4,
    num_sample_img_per_run=2,
    num_sample_patch_per_img=10,
    min_cap_of_patches=1000,
    max_cap_of_patches=8000)

# Start all processes for feeding the patch queue
test_reader.start_feeding_q()

# Fecth one batch from the patche queue
sample_batch = test_reader.next_batch()

# Stop all processes 
test_reader.stop_feeding_q()



