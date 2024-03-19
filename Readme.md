Team Grey<br>
ADL Group coursework
---
Self-supervised pre-training for segmentation

1. Select a self-supervision segmentation algorithm (e.g. masked autoencoder, contrastive learning algorithm, etc). Justify choice.
2. Find large image dataset(s) to generate a pre-trained model, justify choices. 
3. Perform the self-supervised training using the large dataset. 
4. Design and implement a fine-tuning method using a subset of the Pet data set ('The Oxford-IIIT Pet Dataset').
https://www.robots.ox.ac.uk/~vgg/data/pets/
This is a "37 category pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation."
'Cats and Dogs' Parki et al., IEEE Conference on Computer Vision and Pattern Recognition, 2012
"We investigate the fine grained object categorization problem of determining the breed of animal from an image. To this end we introduce a new annotated dataset of pets, the Oxford-IIIT-Pet dataset, covering 37 different breeds of cats and dogs. The visual problem is very challenging as these animals, particularly cats, are very deformable and there can be quite subtle differences between the breeds. We make a number of contributions: first, we introduce a model to classify a pet breed automatically from an image. The model combines shape, captured by a deformable part model detecting the pet face, and appearance, captured by a bag-of-words model that describes the pet fur. Fitting the model involves automatically segmenting the animal in the image. Second, we compare two classification approaches: a hierarchical one, in which a pet is first assigned to the cat or dog family and then to a breed, and a flat one, in which the breed is obtained directly. We also investigate a number of animal and image orientated spatial layouts. These models are very good: they beat all previously published results on the challenging ASIRRA test (cat vs dog discrimination). When applied to the task of discriminating the 37 different breeds of pets, the models obtain an average accuracy of about 59%, a very encouraging result considering the difficulty of the problem."
https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf
5. Design and conduct experiments for network comparisons, at least:
   5(a). Compare the framework with a baseline model trained on the same fine-tuning data, using fully supervised methods.
   5(b).Compare the benefit of the pre-trained segmentation model, using different fine-tuning data set sizes.
6. Describe implemented methods and conduct experiments.
7. Summarise obtained results and draw conclusions.