import os
import glob

from PIL import Image
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
import progressbar
from skimage.color import rgb2hsv, hsv2rgb
import tensorflow_datasets as tfds

def plot_images(ims, num, filename):
    """
    Given a list of (equally-shaped) images
    Save them as `filename` in a `num`-by-`num` square
    """
    if np.min(ims) < 0:
        ims = (np.array(ims) + 1) / 2
    if len(ims) < num ** 2:
        indices = np.arange(len(ims))
    else:
        indices = np.arange(num ** 2)
    image_height, image_width, image_channels = ims.shape[1], ims.shape[2], ims.shape[3]

    full_im = np.zeros((num * image_height, num * image_width))
    
    for index, ims_idx in enumerate(indices):
        column = index % num
        row = index // num
        this_image = ims[ims_idx]
        if len(this_image.shape) == 2:  # No channels greyscale
            this_image = np.reshape((*this_image.shape, 1))
        if this_image.shape[2] == 1:  # One channel greyscale
            this_image = np.reshape(this_image, (image_height, image_width))
            
        full_im[
            row * image_height : (row + 1) * image_height,
            column * image_width : (column + 1) * image_width,
        ] = this_image

    im = Image.fromarray(np.array(full_im * 127.5 + 127.5, dtype=np.uint8), mode='L')
    im.save(filename)
    im.close()



def load_image_files(data_dir, reshape=None, flip=True):
    FILEPATHS = glob.glob(os.path.join(data_dir, "*.jpg"))
    FILEPATHS += glob.glob(os.path.join(data_dir, "*.png"))
    FILEPATHS += glob.glob(os.path.join(data_dir, "*.webp"))
    num = len(FILEPATHS) * (2 if flip else 1)
    images = np.zeros((num, *reshape, 3))
    counter = 0
    for image_path in progressbar.progressbar(FILEPATHS):
        im = Image.open(image_path)
        width, height = im.size
        if reshape is not None:
            im = im.resize(reshape)
        im = np.array(im) / 255.0
        if len(im.shape) == 2:  # No channels greyscale
            im = np.reshape((*im.shape, 1))
        if im.shape[2] == 1:  # One channel greyscale
            im = np.tile(im, (1, 1, 3))
        im = im * 2 - 1
        images[counter] = im
        counter += 1
        if flip:
            images[counter] = im[:, ::-1]
            counter += 1
    np.random.shuffle(images)
    return images


def get_smallnorb_data():
    
    x_train, y_train = tfds.as_numpy(tfds.load(
                          'smallnorb',
                          split='train', 
                          batch_size=-1, 
                          as_supervised=True,
                      ))
    x_test, y_test = tfds.as_numpy(tfds.load(
                          'smallnorb',
                          split='test', 
                          batch_size=-1, 
                          as_supervised=True,
                      ))

    '''
    Class 0 - four-legged animals
    Class 1 - human figures
    Class 2 - airplanes
    Class 3 - trucks
    Class 4 - cars
    '''
    class_label_list = [4]   # Use for filtering dataset according to class label     
    if class_label_list:
        
        uniq_levels = np.unique(y_train)
        uniq_counts = {level: sum(y_train == level) for level in uniq_levels}
        
        # find observation index of each class levels
        groupby_levels = {}
        for ii, level in enumerate(uniq_levels):
            obs_idx = [idx for idx, val in enumerate(y_train) if val == level]
            groupby_levels[level] = obs_idx

        copy_idx = []
        for class_label in class_label_list:
          copy_idx = copy_idx + groupby_levels.get(class_label)

        x_train = x_train[copy_idx]

        
        uniq_levels = np.unique(y_test)
        uniq_counts = {level: sum(y_test == level) for level in uniq_levels}
        
        # find observation index of each class levels
        groupby_levels = {}
        for ii, level in enumerate(uniq_levels):
            obs_idx = [idx for idx, val in enumerate(y_test) if val == level]
            groupby_levels[level] = obs_idx

        copy_idx = []
        for class_label in class_label_list:
          copy_idx = copy_idx + groupby_levels.get(class_label)

        x_test = x_test[copy_idx]

    # normalization
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5
    
    return ((x_train.astype(np.float32), y_train.astype(np.float32)),
            (x_test.astype(np.float32), y_test.astype(np.float32)))


def upscale(im):
    new_im = np.zeros((2 * im.shape[0], 2 * im.shape[1], *im.shape[2:]))
    new_im[::2, ::2] = im
    new_im[1::2, ::2] = im
    new_im[::2, 1::2] = im
    new_im[1::2, 1::2] = im
    return new_im


def plot_interpolation_gif(start,
            finish,
            directory,
            encoder=lambda x: x,
            decoder=lambda x: x,
            frames=120,
            interpolation='linear',
            upscale_times=2):
    if not os.path.exists(directory):
        os.makedirs(directory)
    start_encoded = encoder(np.array([start]))[0]
    finish_encoded = encoder(np.array([finish]))[0]
    if len(start.shape) == 2:
        start = decoder(start)
        finish = decoder(finish)
    if interpolation == 'linear':
        space = np.linspace(start_encoded, finish_encoded, frames)
    elif interpolation == 'sigmoid':
        proportion_start = np.linspace(np.zeros(start_encoded.shape),
                                       np.ones(start_encoded.shape) * np.pi,
                                       frames)
        proportion_start = (np.cos(proportion_start) + 1) / 2
        space = (proportion_start * start_encoded
                 + (1-proportion_start) * finish_encoded)
    interpolated = decoder(space)
    start_upscaled = start
    finish_upscaled = finish
    for i in range(upscale_times):
        interpolated = np.array([upscale(x) for x in interpolated])
        start_upscaled = upscale(start_upscaled)
        finish_upscaled = upscale(finish_upscaled)
    plot_images([start_upscaled], 1, os.path.join(directory, f"start.png"))
    plot_images([finish_upscaled], 1, os.path.join(directory, f"finish.png"))
    for index, im in enumerate(interpolated):
        this_filename = os.path.join(directory, f"interp.{index:05d}.png")
        plot_images([im], 1, this_filename)


    os.system(
        f"ffmpeg -y -r 20 -i {os.path.join(directory, 'interp.%05d.png')}"
        f" -crf 15 {os.path.join(directory, 'interp.mp4')}"
    )

def plot_interpolation_grid(starts,
            finishes,
            filename,
            encoder=lambda x: x,
            decoder=lambda x: x,
            steps=7,
            interpolation='linear',
            upscale_times=2):
    starts_encoded = encoder(starts)
    finishes_encoded = encoder(finishes)
    ims = []
    for i in range(len(starts)):
        start = starts[i]
        if len(start.shape) != 3:
            start = decoder(start.reshape((1, -1)))[0]
        finish = finishes[i]
        if len(finish.shape) != 3:
            finish = decoder(finish.reshape((1, -1)))[0]
        start_encoded = starts_encoded[i]
        finish_encoded = finishes_encoded[i]
        space = np.linspace(start_encoded, finish_encoded, steps)
        images = decoder(space)
        for __ in range(upscale_times):
            start = upscale(start)
            finish = upscale(finish)
        ims.append(start)
        for image in images:
            for __ in range(upscale_times):
                image = upscale(image)
            ims.append(image)
        ims.append(finish)
    plot_images(ims, steps+2, filename)


def build_directories(base_dir, experiment):
    dirs = {}
    dirs["results"] = base_dir
    dirs["experiment"] = os.path.join(base_dir, experiment)
    dirs["training"] = os.path.join(dirs["experiment"], 'training')
    dirs["generated"] = os.path.join(dirs["training"], 'generated')
    dirs["generated_random"] = os.path.join(dirs["generated"], 'random')
    dirs["generated_fixed"] = os.path.join(dirs["generated"], 'fixed')
    dirs["reconstructed"] = os.path.join(dirs["training"], 'reconstructed')
    dirs["reconstructed_random"] = os.path.join(dirs["reconstructed"], 'random')
    dirs["reconstructed_fixed"] = os.path.join(dirs["reconstructed"], 'fixed')
    dirs["output"] = os.path.join(dirs["experiment"], 'output')
    dirs["output_models"] = os.path.join(dirs["output"], 'models')
    dirs["output_interpolations"] = os.path.join(dirs["output"], 'interpolations')

    for __, i in dirs.items():
        if not os.path.exists(i):
            os.makedirs(i)
    return dirs

'''
def plot_loss(df):  
	plt.plot(df['Gx'], label='generator loss for images')
	plt.plot(df['Gz'], label='generator loss for latent vectors')
	plt.plot(df['Dx'], label='adversarial loss for images')
	plt.plot(df['Dz'], label='adversarial loss for latent vectors')
	plt.plot(df['Rx'], label='image reconstruction loss')
	plt.plot(df['Rz'], label='latent vector reconstruction loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
	plt.show()

def get_all_loss_plots():
	# This code is not automated to get plots for different types of loss
	# Adding this code for completeness
	# Collect and process all the console logs after the model has been trained
	# Create a csv file with loss values collected from console logs and give following column names
	# Gx,Gz,Dx,Dz,Rx,Rz
	df = pd.read_csv('classfour.csv')
	plot_loss(df)
'''
