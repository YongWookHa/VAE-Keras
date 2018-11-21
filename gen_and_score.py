import sys
import os
import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import scipy.misc

from inception_score import get_inception_score
INCEPTION_SCORE = 0

if __name__ == '__main__':
    vae_json_path = sys.argv[1]
    vae_w_path = sys.argv[2]
    num = int(sys.argv[3])
    print("--------------------------------")
    print('vae_json_path : ', vae_json_path)
    print('vae_w_path : ', vae_w_path)
    print('num : ', num)
    print("--------------------------------")

    with open(vae_json_path, 'r') as f:
        vae = model_from_json(f.read())
    vae.load_weights(vae_w_path)

    folder = os.path.basename(os.path.dirname(vae_json_path))

    batch_size = 64
    os.makedirs('samples/%s/' % folder, exist_ok=True)
    for b in range(num//64):
        print(batch_size*b+1, '/',num//64)
        noise = np.random.normal(0, 1, (batch_size, 512))
        imgs = vae.predict(noise)
        imgs = (0.5 * imgs + 0.5) * 255
        for i in range(batch_size):
            scipy.misc.toimage(imgs[i], cmin=0.0, cmax=255).save('samples/%s/%d.jpg' % (folder, b*batch_size+i))
    
    rest = num % batch_size
    noise = np.random.normal(0, 1, (rest, 512))
    imgs = vae.predict(noise)
    imgs = (0.5 * imgs + 0.5) * 255
    for i in range(rest):
        scipy.misc.toimage(imgs[i], cmin=0.0, cmax=255).save('samples/%s/%d.jpg' % (folder, num-rest+i))

    
    if INCEPTION_SCORE == 1:
        li = os.listdir('samples/%s/' % folder)
        for filename in li:
            if filename[-4:] == '.jpg':
                img = scipy.misc.imread(os.path.join(path, filename))
                imgs.append(img)
            else:
                pass
        
        result = get_inception_score(imgs)

        with open('samples/%s/inception_score.txt' % (folder), 'w') as f:
            f.write('inception score...\n')
            f.write('mean : %s\n' % result[0])
            f.write('std : %s\n' % result[1])
            f.close()
            
        print('scoring finished!')
        print('mean : %s\n' % result[0])
        print('std : %s\n' % result[1])
