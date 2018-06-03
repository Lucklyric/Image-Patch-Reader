"""
Created on: 2018-Jun-02
File: data_reader_mp.py
'''
DataReader by multiprocessing
@author: Alvin(Xinyao) Sun
"""

import glob
import multiprocessing
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread


class QThread(threading.Thread):
    def __init__(self, name, target):
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


class BytesChannel(object):
    def __init__(self, maxsize):
        """Shared memory wrapper
        Args:
            maxsize (int): length of array
        """
        # Float buffer for images
        self.buffer_data = multiprocessing.RawArray(np.ctypeslib.ctypes.c_ubyte, maxsize * 4 + 1)
        self.buffer_len = multiprocessing.Value("i")
        self.checking = multiprocessing.Value("i")
        self.empty = multiprocessing.Semaphore(1)
        self.full = multiprocessing.Semaphore(0)
        self.checking.value = 0

    def send(self, data):
        """ Save data to shared memory
        
        Args:
            data (RawArray): BytesArray
        """
        self.empty.acquire()
        nitems = len(data)
        self.buffer_len.value = nitems
        self.buffer_data[:nitems] = data
        self.full.release()

    def recv(self):
        """ Receive data from shared memory
        
        Returns:
            data : BytesArray 
        """

        self.full.acquire()
        data = self.buffer_data[:self.buffer_len.value]
        self.empty.release()
        return data


class DataReaderPatchWiseMP(object):
    def __init__(self,
                 path,
                 batch_size=64,
                 patch_size=64,
                 num_sample_img_per_run=2,
                 num_sample_patch_per_img=2,
                 num_process=6,
                 min_cap_of_patches=2000,
                 max_cap_of_patches=8000,
                 verbose=True):
        """ DataReader multi-process version
        
        Args:
            path (string): path for image folder 
            batch_size (int, optional): Defaults to 64. 
            patch_size (int, optional): Defaults to 64. 
            num_sample_img_per_run (int, optional): Defaults to 2. 
            num_sample_patch_per_img (int, optional): Defaults to 2. 
            num_process (int, optional): Defaults to 6. 
            min_cap_of_patches (int, optional): Defaults to 2000. 
            max_cap_of_patches (int, optional): Defaults to 8000. 
            verbose (bool, optional): Defaults to True. 
        """

        self.data_q = []
        self.files = glob.glob(path, recursive=True)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_sample_img_per_run = num_sample_img_per_run
        self.num_sample_patch_per_img = num_sample_patch_per_img
        self.lock = threading.Lock()
        self.q_process = []
        self.min_cap_of_patches = min_cap_of_patches
        self.max_cap_of_patches = max_cap_of_patches
        self.shuffle_period = -1
        self.verbose = verbose
        self.fetch_thread = QThread("fetch", target=self.fetching_loop)
        self.total_sample = self.num_sample_img_per_run * self.num_sample_patch_per_img
        self.bytes_channel = BytesChannel(self.total_sample * patch_size * patch_size)
        self.shuffle_period = -1
        self.shuffle_check_len = -1
        for i in range(num_process):
            self.q_process.append(multiprocessing.Process(target=self.run_loop, args=([self.bytes_channel])))

    def append_to_q(self, imgs):
        """ Append data to data queue
        
        Args:
            imgs (numpy array): patches
        """

        self.lock.acquire()
        self.data_q.extend(imgs)
        self.lock.release()
        if self.verbose:
            print("[%s]: Append data from_thread %s, current Q len=%d" % (self.__class__.__name__, threading.current_thread().getName(), len(self.data_q)))
        # print(threading.current_thread().running_state())
        return

    def pop_from_q(self):
        """Pop one batch from data queue
        
        Returns:
            batch: one batch of patches
        """

        while True:
            if len(self.data_q) > self.min_cap_of_patches:
                if self.verbose:
                    print("[%s]: %d left in Q" % (self.__class__.__name__, len(self.data_q)))
                break
            # time.sleep(1 / 120.0)
        self.lock.acquire()
        if self.shuffle_period == -1 or self.shuffle_period > (self.shuffle_check_len / self.batch_size) - 1:
            np.random.shuffle(self.data_q)
            self.shuffle_period = 0
            self.shuffle_check_len = len(self.data_q)
            print("[%s]: shuffle Q" % self.__class__.__name__)
        batch_imgs = self.data_q[:self.batch_size]
        del self.data_q[:self.batch_size]
        self.lock.release()
        self.shuffle_period += 1
        print("[%s]: shuffle_period=%d" % (self.__class__.__name__, self.shuffle_period))
        return batch_imgs

    def fetching_loop(self):
        """A child thread for fetching data from child process
        """

        t = threading.current_thread()
        while t.running_state():
            if len(self.data_q) < self.max_cap_of_patches:
                self.bytes_channel.checking.value = 0
                imgs = self.bytes_channel.recv()
                imgs = np.reshape(np.frombuffer(bytearray(imgs), dtype=">f4").astype(float), [self.total_sample, self.patch_size, self.patch_size, 1])
                self.append_to_q(imgs)
            else:
                self.bytes_channel.checking.value = 1
            # time.sleep(1 / 30.0)

    def run_loop(self, ch):
        """Child process target loop
        
        Args:
            ch (ByteArray): Python ByteArray with shared memory 
        """

        while True:
            if ch.checking.value == 0:
                try:
                    imgs = self.prepare_data()
                    ch.send(bytearray(imgs.flatten().astype('>f4')))
                except ValueError:
                    print("value Error")
            else:
                time.sleep(1)  # save cpu usage
            time.sleep(1 / 120.0)

    def prepare_data(self):
        """Main function for extracting patches
        
        Returns:
            numpy assary: images patches
        """

        img_patches = []
        file_idxes = np.random.randint(0, len(self.files), size=self.num_sample_img_per_run)
        for f_idx in file_idxes:
            # Read all imgs
            img = imread(self.files[f_idx], as_grey=True)
            img = self.transform(img)
            [h, w] = np.shape(img)
            rows = np.random.randint(0, h - self.patch_size - 1, size=self.num_sample_patch_per_img)
            cols = np.random.randint(0, w - self.patch_size - 1, size=self.num_sample_patch_per_img)
            for k in range(self.num_sample_patch_per_img):
                k_h = rows[k]
                k_w = cols[k]
                img_patches.append(img[k_h:k_h + self.patch_size, k_w:k_w + self.patch_size])
        return np.asarray(img_patches)

    @staticmethod
    def transform(img):
        """Original image transform function
        
        Args:
            img (numpy array): original image
        
        Returns:
            numpy array: transformed image
        """

        scale = np.random.choice([256, 384, 512], 1)
        img = resize(img, [scale, scale])
        return img

    def next_batch(self):
        """Fetching next batch from data queue
        
        Returns:
            numpy array: one batch of patches
        """

        batch = self.pop_from_q()
        imgs = np.reshape(batch[0], [-1, self.patch_size, self.patch_size, 1])
        return imgs

    def start_feeding_q(self):
        """Start all child processes and child thread
        """

        for i in range(len(self.q_process)):
            np.random.seed(int(time.time()) + i)
            self.q_process[i].start()
        self.fetch_thread.start()

    def stop_feeding_q(self):
        """Stop all child process and child thread
        """

        for process in self.q_process:
            process.terminate()
        self.fetch_thread.stop_thread()


# For testing
if __name__ == '__main__':
    test_reader = DataReaderPatchWiseMP(
        path="/train/**/*.JPEG",  # path for your image folder
        batch_size=64,
        patch_size=64,
        num_process=4,
        num_sample_img_per_run=2,
        num_sample_patch_per_img=10,
        min_cap_of_patches=1000,
        max_cap_of_patches=8000)
    test_reader.start_feeding_q()
    sample_batch = test_reader.next_batch()
    print("shape of sample batch" + str(sample_batch.shape))
    plt.imsave("test.png", np.reshape(sample_batch[0], [64, 64]), cmap="gray")
    # while True:
    #     sample_batch = test_reader.next_batch()
    #     time.sleep(0.5)
    print("Done")
