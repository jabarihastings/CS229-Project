Here is a useful dataset with TONS of images:
https://github.com/cabani/MaskedFace-Net
(uses the data set: https://arxiv.org/pdf/2008.08016.pdf)

This Kaggle dataset (https://www.kaggle.com/andrewmvd/face-mask-detecti ) contains 853 images belonging to the 3
classes, as well as their bounding boxes in the PASCAL VOC format.The classes are 1) With mask, 2) Without mask, and
3) Mask worn incorrectly. Notice that "With mask" refers to wearing mask correctly.
The images (with multiple people and thus multiple lables) are cropped and placed into the respective folders using an
adoption of "pascalvoc-to-image" (https://gitlab.com/straighter/pascalvoc-to-image). There exists some differences in
the XML formatting of the PASCAL VOC annotations, and thus changes were needed to be made in order to make the
pascalvoc-to-image script work. This resulted in 3232 images of "With mask", 124 images of "mask worn incorrectly", and
717 images of "without mask".

Requirements:
We recommend using python3 and a virtual env. Run the follow commands:
    virtualenv -p python3 .env
    source .env/bin/activate
    pip install -r code/requirements.txt
