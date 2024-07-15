import matplotlib.pyplot as plt
import numpy as np

def plot_imgs(generator):
    batch, _ = next(generator)  
    plt.figure(figsize=(15,5))
    for i, img in zip(range(1, 6), batch):
        plt.subplot(1, 5, i)
        plt.imshow(img)
    plt.show()

def plot_rec_imgs(generator, number_plots, autoencoder, verbose=0):
    images, _ = next(generator)
    plt.figure(figsize=(8,8))
    for i in range(number_plots):    
        plt.title('train data')
        plt.subplot(1,3,1)
            
        # ORIGINAL IMAGE
        original_img = images[i]
        plt.title('Original')
        plt.imshow(original_img)
        plt.xticks([])
        plt.yticks([]) 
            
        # RECONSTRUCTED IMAGE
        expand_img = np.expand_dims(original_img, axis=0)
        decoded_img = autoencoder.predict(expand_img, verbose=verbose)
        loss = autoencoder.evaluate(decoded_img, expand_img, verbose=verbose)
        decoded_img = np.squeeze(decoded_img, axis=0)
        plt.subplot(1,3,2)
        plt.title(f'Reconstructed loss={loss:0.3f}')
        plt.imshow(decoded_img)           
        plt.xticks([])
        plt.yticks([]) 
        plt.show()
            
def plot_loss_history(history):
    plt.rcParams["figure.figsize"] = (10,6)
    loss = history.history['loss'] 
    val_loss = history.history['val_loss'] 
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, '-o', label='Training loss')
    plt.plot(epochs, val_loss, '-o', label='Validation loss')
    plt.title('Training and valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def get_error(generator, autoencoder, batch_size, verbose=0):
    recon_error_list = []
    if (generator.samples >= batch_size):
        for _ in range(0, generator.samples // batch_size + 1):
            batch, _ = next(generator)
            for img in batch:
                reconstruction_error = autoencoder.evaluate(
                    autoencoder.predict(np.expand_dims(img, axis=0), verbose=verbose),
                    np.expand_dims(img, axis=0),
                    verbose=verbose)
                recon_error_list.append(reconstruction_error)
    else:
        batch, _ = next(generator)
        for img in batch:
            reconstruction_error = autoencoder.evaluate(
                    autoencoder.predict(np.expand_dims(img, axis=0), verbose=verbose),
                    np.expand_dims(img, axis=0),
                    verbose=verbose)
            recon_error_list.append(reconstruction_error)
    recon_error_list = np.array(recon_error_list)  
    return recon_error_list

def plot_rec_distribution(clean, fraud, threshold):
    fig, ax = plt.subplots(figsize=(10,10))

    ax.hist(clean, bins=50, density=True, label="clean", alpha=.3, color="green")
    ax.hist(fraud, bins=50, density=True, label="fraud", alpha=.6, color="red")
    plt.axvline(x=threshold, color= 'b', label="treshold", linestyle='--', linewidth=3)
    plt.xlabel('Reconstruction error')
    plt.ylabel('Count')

    plt.title("Distribution of the Reconstruction Loss")
    plt.legend()
    plt.show()
    
def get_classification_metrics(clean, fraud, threshold):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    for i in range(0, len(clean)):
        test_reconstruction_error = clean[i]
        if test_reconstruction_error > threshold:
            false_negative += 1    
        else:
            true_positive += 1
            
    for i in range(0, len(fraud)):
        test_reconstruction_error = fraud[i]
        if test_reconstruction_error > threshold:
            true_negative += 1    
        else:
            false_positive += 1
            
    return true_positive, true_negative, false_positive, false_negative

def print_classification_metrics(true_positive, true_negative, false_positive, false_negative):
    len = true_positive + true_negative + false_positive + false_negative
    precisicon = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    accuracy = (true_positive + true_negative) / len
    
    # print(f'Количество примеров: {len}')
    # print(f'false negative {false_negative}')
    # print(f'false positive {false_positive}')
    # print(f'true negative {true_negative}')
    # print(f'true positive {true_positive}')

    print(f'precisicon {precisicon:0.3f}')
    print(f'recall {recall:0.3f}')
    print(f'F {F:0.3f}')
    print(f'accuracy: {accuracy:0.3f}')