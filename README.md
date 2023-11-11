# CVMS_Classification

The process involves two main steps:

Step 1: Annotator

    1.1 Using VGG software  (https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to label 19 points around the cervical vertebrae (C2, C3, and C4).

    1.2 Confirming the position of these 19 points on each image using view.py.

    1.3 Converting these points into relevant features, such as size, shape, and concavity, through pointTofeature.py.

Step 2: Classification (70% for training, 30% for testing)

    2.1 Categorizing into five groups: 3 Single rater categories (cvms_classification_SingleRater.py) and 2 Voting system categories, including Complete agreement and Majority agreement (cvms_classification_Voting.py).

    2.2 Implementing 4 Single models and 2 Ensemble models.

    2.3 Utilizing 10-fold Cross-validation, Feature selection, and Hyperparameter tuning to optimize each model's parameters.

    2.4 Considering Model generalization by random selecting 30 percent of each dataset to maintain the same distribution.
