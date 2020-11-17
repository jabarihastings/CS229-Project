# Face Mask Detection with Machine Learning

In this project, we explore a host of learning algorithms in an attempt to create a classifier that is able to discern whether a human is 1) wearing a mask correctly, 2) wearing a mask incorrectly, and 3) not wearing a mask. 

#### Datasets
We make use of a subset of the images made avaialable through the following datasets
- Flickr-Faces-HQ Dataset (FFHQ): The dataset consists of 70,000 high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background.
- MaskedFaceNet:  This dataset consists of human faces with a correctly or incorrectly worn mask (137,016 images) based on the dataset Flickr-Faces-HQ (FFHQ)
- Kaggle Face Mask Detection: This dataset consists of 850 images of people wearing real masks correctly and incorrectly of people not weaing masks. The images do not come in a uniform size and do not contain a single subject, but annotation is provided for separation.

We combined a subset of FFHQ and MaskedFaceNet datasets to get a corpus (with ~4000 images) of people in all three categories. We used the entire Kaggle Face Maske Detection dataset as a second resource for testing and training. Our datasets can be found at https://stanford.app.box.com/folder/126299596483.

### Licenses
**In the following the licenses of the original FFHQ-dataset**: The individual images were published in Flickr by their respective authors under either Creative Commons BY 2.0, Creative Commons BY-NC 2.0, Public Domain Mark 1.0, Public Domain CC0 1.0, or U.S. Government Works license. All of these licenses allow free use, redistribution, and adaptation for non-commercial purposes. However, some of them require giving appropriate credit to the original author, as well as indicating any changes that were made to the images. The license and original author of each image are indicated in the metadata.

- https://creativecommons.org/licenses/by/2.0/
- https://creativecommons.org/licenses/by-nc/2.0/
- https://creativecommons.org/publicdomain/mark/1.0/
- https://creativecommons.org/publicdomain/zero/1.0/
- http://www.usa.gov/copyright.shtml

The dataset itself (including JSON metadata, download script, and documentation) is made available under Creative Commons BY-NC-SA 4.0 license by NVIDIA Corporation. You can use, redistribute, and adapt it for non-commercial purposes, as long as you a) give appropriate credit by citing our paper, b) indicate any changes that you've made, and c) distribute any derivative works under the same license.

- https://creativecommons.org/licenses/by-nc-sa/4.0/

**In the following the licenses of the MaskedFace-Net dataset**: The dataset is made available under Creative Commons BY-NC-SA 4.0 license by NVIDIA Corporation. You can use, redistribute, and adapt it for non-commercial purposes, as long as you:
i. give appropriate credit by citing our papers: 
>Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, Elsevier, 2020. [accepted, arXiv:2008.08016](https://arxiv.org/pdf/2008.08016.pdf)
 
> Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 2020, DOI:10.32604/cmes.2020.011663

ii. indicate any changes that you've made,
iii. and distribute any derivative works under the same license. https://creativecommons.org/licenses/by-nc-sa/4.0/

**In the following the licenses of the original FFHQ-dataset**: The individual images were published in Flickr by their respective authors under Public Domain CC0 1.0, or U.S. Government Works license. This license allows free use, redistribution, and adaptation for non-commercial purposes.

- https://creativecommons.org/publicdomain/zero/1.0/

### Usage 
Requirements:
We recommend using python3 and a virtual env. Run the follow commands:
```sh
$ virtualenv -p python3 .env
$ source .env/bin/activate
$ pip install -r code/requirements.txt
```

### Todos
 - Add instructions for how to run code
 - Add Night Mode

# Acknowlegements
We would like to thank Tero Karras, Samuli Laine, and Timo Aila for their work on face detection, which resulted in the FFHQ dataset
>A Style-Based Generator Architecture for Generative Adversarial Networks
>Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)
>https://arxiv.org/abs/1812.04948

We also would like to thank Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi for their work on face mask detection, which resulted in the MaskedFaceNet dataset
>Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, Elsevier, 2020. [accepted, arXiv:2008.08016](https://arxiv.org/pdf/2008.08016.pdf)

>Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 2020, DOI:10.32604/cmes.2020.011663
