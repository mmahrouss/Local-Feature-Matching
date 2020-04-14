# Local-Feature-Matching
*Local Feature Matching: Computer Vision University Project*

This project aims at learning to generate image features around a local point in the image. 

The project consists of three parts in `student.py`:

1. Generating Interest Points with Harris `get_interest_points`
2. Generating SIFT-like features around each interest point `get_features`
3. Matching Features between two images `match_features`

## Results:
![](images/notre_dame_matches.jpg
)
Matches after Whole pipeline on Notre Dame. Matches: 1113 Accuracy on 50most confident: 100% Accuracy on 100 most confident: 99% Accuracy on all matches:75%
![](images/mt_rushmore_matches.jpg
)
Matches after Whole pipeline on Mount Rushmore . Matches: 55 Accuracy on50 most confident: 94% Accuracy on all matches: 92%
![](images/e_gaudi_matches.jpg
)
Matches after Whole pipeline on Epicopal Gaudi . Matches: 13 Accuracy onall matches: 23%


