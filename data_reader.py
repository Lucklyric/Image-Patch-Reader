"""
Created on 2018-03-23
Project: Image-Patch-Reader
File: DataReader
...
@author: Alvin(Xinyao) Sun
"""
import numpy as np
import matplotlib.pyplot as plt
# from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
import glob

import threading
import time


class QThread(threading.Thread):
    def __init__(self, name, target):
        """
             python threading wrapper
            :param self:
            :param name: name of the thread
            :param target: target function
        """
        threading.Thread.__init__(self, name=name, target=target)
        self.running = True
        self.run_lock = threading.Lock()

    def stop_thread(self):
        self.run_lock.acquire()
        self.running = False
        self.run_lock.release()

    def running_state(self):
        self.run_lock.acquire()
        state = self.running
        self.run_lock.release()
        return state


class DataReaderPatchWise(object):
    def __init__(self,
                 path,
                 batch_size=64,
                 patch_size=64,
                 num_thread=2,
                 num_image_per_sample=64,
                 num_patches_per_image=4,
                 min_cap_of_patches=6000,
                 max_cap_of_patches=8000):
        """
        Data Reader Instance
            :param path: path of the folder
            :param batch_size=64:  size of the batch
            :param patch_size=64:  size of the patch
            :param num_thread=2:  number of threads
            :param num_image_per_sample=2:  number of sampled images
            :param num_patcehs_per_image=2:  number of patches per sample
            :param min_cap_of_patches=6000:
            :param max_cap_of_patches=8000:
        """
        self.data_q = []
        self.path = path
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.lock = threading.Lock()
        self.q_threads = []
        self.min_cap_of_patches = min_cap_of_patches
        self.max_cap_of_patches = max_cap_of_patches
        self.num_image_per_sample = num_image_per_sample
        self.num_patches_per_image = num_image_per_sample
        self.shuffle_period = -1
        self.files = glob.glob(path, recursive=True)
        for i in range(num_thread):
            q_thread = QThread(name="q_" + str(i), target=self.run_loop)
            self.q_threads.append(q_thread)

    def append_to_q(self, imgs=None):
        """
        Append pathces to the queue
            :param self:
            :param imgs=None:
        """
        self.lock.acquire()
        self.data_q.extend(imgs)
        print(len(self.data_q))
        self.lock.release()
        return

    def pop_from_q(self):
        """
        Pop patcehs from the queue
            :param self:
        """
        while True:
            check = self.batch_size
            if self.shuffle_period == -1:
                check = self.min_cap_of_patches
            if len(self.data_q) > check:
                print("[%s]: %d left in Q" % (self.__class__.__name__, len(self.data_q)))
                # self.lock.acquire()
                # np.random.shuffle(self.data_q)
                # self.lock.release()
                break
            time.sleep(1 / 30.0)
        self.lock.acquire()
        if self.shuffle_period == -1 or self.shuffle_period > (self.min_cap_of_patches / self.batch_size) - 1:
            np.random.shuffle(self.data_q)
            self.min_cap_of_patches = len(self.data_q)
            self.shuffle_period = 0
            print("[%s]: shuffle Q" % self.__class__.__name__)
        batch = self.data_q[:self.batch_size]
        del self.data_q[:self.batch_size]
        self.shuffle_period += 1
        print("[%s]: shuffle_period=%d" % (self.__class__.__name__, self.shuffle_period))

        self.lock.release()
        return batch

    def run_loop(self):
        """
        Running loop function
            :param self:
        """
        t = threading.current_thread()
        while t.running_state():
            while len(self.data_q) < self.max_cap_of_patches:
                if t.running_state() is False:
                    return
                try:
                    img_batches = self.prepare_data()
                    self.append_to_q(img_batches)
                except ValueError:
                    print("value Error")

            time.sleep(1 / 30.0)

    def prepare_data(self):
        """
        Reading randomly selected images and extract patches randomly
            :param self:
        """
        file_idxes = np.random.randint(0, len(self.files), size=[self.num_image_per_sample])
        patches = []
        sample_size = self.num_patches_per_image
        for i in range(len(file_idxes)):
            img = imread(self.files[file_idxes[i]],as_grey=True)
            img = resize(img, [256, 256])
            [h, w] = np.shape(img)
            rows = np.random.randint(0, h - self.patch_size - 1, size=sample_size)
            cols = np.random.randint(0, w - self.patch_size - 1, size=sample_size)
            for k in range(sample_size):
                k_h = rows[k]
                k_w = cols[k]
                patches.append(img[k_h:k_h + self.patch_size, k_w:k_w + self.patch_size])
        # np.random.shuffle(patches)
        return patches

    def next_batch(self):
        """
        Call to get a batch of patches
            :param self:
        """
        return np.reshape(self.pop_from_q(), [-1, self.patch_size, self.patch_size, 1])

    def start_feeding_q(self):
        """
        Start all threads
            :param self:
        """
        for thread in self.q_threads:
            thread.start()

    def stop_feeding_q(self):
        """
        Terminate all threads
            :param self:
        """
        for thread in self.q_threads:
            thread.stop_thread()


# For testing
if __name__ == '__main__':
    test_reader = DataReaderPatchWise(
        path="/home/alvinsun/Documents/Data/imagenet/ILSVRC/Data/DET/train/**/*.JPEG",
        batch_size=64,
        patch_size=64,
        num_thread=2,
        num_image_per_sample=1,
        num_patches_per_image=2,
        min_cap_of_patches=1000,
        max_cap_of_patches=8000,
        )
    test_reader.start_feeding_q()
    sample_batch = test_reader.next_batch()
    print("shape of sample batch" + str(sample_batch.shape))
    plt.imshow(np.reshape(sample_batch[0], [64, 64]), cmap="gray")
    time.sleep(100)
    print("Done")