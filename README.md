# video-object-detect-and-tracking

This project takes a set of training images of a particular object or a class of same object (e.g. people) and a video to learn from the image set and detect such object inside the video. And will keep tracking the object inside the video.

We use HOG and SVM to do the object detection. And compute SIFT descriptors and do matching between frames and them compute homography between each frame for tracking.

Before testing or tracking, please provide both positve and negative train dara set ot train a model.
To train the model:
```
py -3 HogSvm.py positive-images-folder-path negative-images-folder-path model-name
```

To test trained model with images:
(merge flag should be set to 0 for no merge and 1 for merge)
```
py -3 test.py test-image-folder-path model-path number-of-test-images merge-flag
```

To do detection and tracking inside the video:
```
py tracking.py video-path save-output-video-path model-path
```
