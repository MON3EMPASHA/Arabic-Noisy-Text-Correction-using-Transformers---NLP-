<!-- Slide number: 1 -->

![IMSA Conference 15-16 July 2023](Picture2.jpg)
Intelligent Methods, Systems, and Applications​
(IMSA-23)
# A Framework for Assessing Physical Rehabilitation Exercises
Moamen Zaher
Faculty of Computer Science
October University for Modern Sciences and Arts (MSA)
Ahmed Samir
Faculty of Physical Therapy
October University for Modern Sciences and Arts (MSA)
Assoc. Prof. Ayman Ezzat Atia
Faculty of Computer Science
October University for Modern Sciences and Arts (MSA)
Dr. Laila M. Abdelhamid
Information System Department
Faculty of Computers &  Artificial Intelligence
Helwan University
Dr. Amr Ghoneim
Computer Science Department
Faculty of Computers & Artificial Intelligence
Helwan University

![](Picture6.jpg)
Paper ID : 168

### Notes:
- Good Morning My Name is ... My paper ID is ... we are proposing " ..title.." in collabration with ....... and under supervision of Dr.. and Dr.. and Dr...  This paper achieves third goal of SDGs which is "Good Health and Well-Being"

<!-- Slide number: 2 -->
# Outlines
Introduction
Motivation
Problem Statement
Related Work
Proposed Framework
Datasets
Expierments
Results
Conclusion

### Notes:
The Agenda for this presentation is as follows

<!-- Slide number: 3 -->

![](Picture2.jpg)
# Introduction 1/2

![Health Benefits of Physiotherapy | Miskawaan Health](Picture2.jpg)
Rehabilitation :
a set of interventions designed to optimize functioning and reduce disability in individuals with health conditions in interaction with their environment”
Rehabilitation -> shortening hospital stays

Rehabilitation is a long-term recovery .
WHO (World health Organization / Health Topics / Rehabilitation[1]

### Notes:
-First Let's talk about Rehabilitations
- What is rehabilitation ? is basically when some one has an accident or a sports man have an injury and he/she has to perform some sort of exercise to recover from this muscle injury  ......
- Rehab aims to ....
- this is a longterm process
-WHO Initiative for 2030

<!-- Slide number: 4 -->
# Introduction 2/2

![Medical rehabilitation physical therapy Royalty Free Vector](Picture2.jpg)
Rehabilitation  is a long process so we need keep track of the progress .
Different skeleton parts , angles and trajectories for different exercises .
Cut-Down Cost of rehabilitation .
Allow Patients to practice exercises at home without the need to go the physio clinics .
Real-Time feedback for the patient whether he’s done the exercise correctly or not .
Allow Doctors to monitor patients progress .
WHO (World health Organization / Health Topics / Rehabilitation

### Notes:
we try to track patient's progress , cut down cost of rehab as patient will perform the exerscie at home and provide real-time feedback

<!-- Slide number: 5 -->
# Motivation

![Figure thumbnail gr4](Picture2.jpg)
Map of leading health conditions requiring rehabilitation in each country, 2019

![Figure thumbnail gr4](Picture2.jpg)
Map of leading health conditions requiring rehabilitation in each country, 2019

Globally, an estimated 2.4 billion people are currently living with a health condition that benefits from rehabilitation.
more than 50% of people do not receive the rehabilitation services they require.
Rehabilitation services are often under funded and under valued, particularly in countries without strong health systems.
<10 per 1 million The number of skilled rehabilitation practitioners in low- and middle-income countries
the number of people over 60 years of age predicted to double by 2050, and more people are living with chronic diseases.
WHO (World health Organization / Health Topics / Rehabilitation

### Notes:
Heatmap of the leading health conditions rqui..
about 2.4 billion perople have health condition , 50% of this number doesn't recieve rehab services due to the lack of the professions

https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)32340-0/fulltext

<!-- Slide number: 6 -->

![](Picture2.jpg)
# Problem Statement
Lack of prioritization, funding, policies and plans for rehabilitation at a national level.
Lack of available rehabilitation services outside urban areas, and long waiting times.
High out-of-pocket expenses and non-existent or inadequate means of funding.
Lack of trained rehabilitation professionals, with less than 10 skilled practitioners per 1 million population in many low- and middle-income settings.
Lack of resources, including assistive technology, equipment and consumables.
The need for more research and data on rehabilitation.
Ineffective and under-utilized referral pathways to rehabilitation.

 WHO (World health Organization / Rehabilitation 10 November 2021

### Notes:
WHO has published several challenges in this domain.we try to address these highlighted challenges

<!-- Slide number: 7 -->
# Related Work

![](ContentPlaceholder4.jpg)
B. Debnath, M. O’Brien, M. Yamaguchi, et al., “A review of computer vision-based approaches for physical rehabilitation and assessment,” Multimedia Systems, vol. 28, no. 2, pp. 209–239, 2022 [2]

### Notes:
This graph represents the CV community contribution in this domain.
Previous work in this domain can be splitted into twp categories ...
We Focus on the first category and the direct rehabilitation sector
Direct Rehab is when the patient perform the rehab exerise phyiscally whie virtual is when the patient play a game that try to address the recovery of the muscle

<!-- Slide number: 8 -->

![](Picture9.jpg)
# Related Work 1/2 : Direct rehabilitation systems  :  Pure vision‐based
RGB or RGB-D camera input.
pilot study to compare the pose estimates produced by four deep models based on RGB input with those of the MS Kinect based on RGB-D data.
choose the MS Kinect when tested with Parkinson’s disease patients in their homes.
capturing exercise information, evaluating patient performance, providing therapeutic feedback to the patient and the therapist, checking the progress of the user over the course of the physical therapy, and supporting the patient throughout this period.

![](Picture5.jpg)

![](Picture7.jpg)
Gu Y, Pandit S, Saraee E, Nordahl T, Ellis T, Betke M. Home-based physical therapy with an interactive computer vision system. InProceedings of the IEEE/CVF International Conference on Computer Vision Workshops 2019 (pp. 0-0).

<!-- Slide number: 9 -->
# Related Work 2/2  : Virtual Rehabilitation systems  :  Multi-Modal

![](Picture5.jpg)
proposes the combination of a multi-label classifier, Circular Classifier Chain (CCC), with a multimodal classifier, Fusion using a Semi-Naive Bayesian classifier (FSNBC).

![](Picture7.jpg)

![page2image526898992](Picture4.jpg)
Rivas JJ, del Carmen Lara M, Castrejon L, Hernandez-Franco J, Orihuela-Espina F, Palafox L, Williams A, Bianchi-Berthouze N, Sucar LE. Multi-label and multimodal classifier for affective states recognition in virtual rehabilitation. IEEE Transactions on Affective Computing. 2021 Feb 1;13(3):1183-94.

<!-- Slide number: 10 -->
# Related Work 1/2 :

![](Picture14.jpg)
This research intended to classify different types of exercises by implementing spike train features into deep learning.
UI-PRMD Dataset
This paper chose to adopt ResNet as their CNN model for classification
Data has been parted into 100 frames
The classification achieved 77%

![](Picture12.jpg)

![](Picture10.jpg)
F. A. Rashid, N. S. Suriani, M. N. Mohd, M. R. Tomari, W. N. W. Zakaria, and A. Nazari, “Deep convolutional network approach in spike train analysis of physiotherapy movements,” in Advances in Electronics Engineering: Proceedings of the ICCEE 2019, Kuala Lumpur, Malaysia, pp. 159–170, Springer Singapore, 2020 [3]

### Notes:
First related work "title"

We chose this work to show how encoding greatly affects the results as we will demonstrate later

worked on the same dataset , tried to visualize the points into a graph and use this graph as an input for CNN (in this case ResNet)

<!-- Slide number: 11 -->
Related Work 2/2 :
Design a new deep learning model by integrating criss-cross attention and edge convolution to extract discriminative features from the skeleton sequence for action recognition
UTD-MHAD and MSR-Action3D Datasets
CNN
The proposed method achieved average accuracies of 99.53% and 95.64% respectively.

![](Picture13.jpg)

![](Picture17.jpg)
N. Tasnim and J.-H. Baek, “Dynamic edge convolutional neural network for skeleton-based human action recognition,” Sensors, vol. 23, no. 2,p. 778, 2023. [4]

### Notes:
Second related work "title"

We chose this work as it have the best recorded results so far .
again worked on the same dataset in addition to another one
Applied CNN .
The paper also include the best recorded results

<!-- Slide number: 12 -->
# Proposed Framework

![](ContentPlaceholder4.jpg)
System Overview

### Notes:
This graph shows our proposed methodology.
The encoding part here is the main contribution the led to the results we will discuss later , as we mentioned in the first paper.

We will explain each part of the graph in details later on

<!-- Slide number: 13 -->
# Datasets
UI-PRMD [5]
10 exercises ,10  individuals repeated 10 times
Vicon optical tracker, and a Kinect camera
The data include the motion measurement for 22 joints
Text Files

Collected Dataset
3 exercises, 1 individual repeated 7 times
Mini squat – Sit to stand – Straight leg raise
RGB camera
Video Files
Extracted 33 body joints

![](Picture6.jpg)

![](Picture2.jpg)

### Notes:
We used 2 datasets. first is our benchmark dataset
second is our own collected dataset by a an expierenced phyisiotherapist

<!-- Slide number: 14 -->
# Preprocessing

![](Picture4.jpg)
The Kinect Camera captures 22 body joints are stored in a vector V.
At each time point t,
The three-dimensional coordinates xt, yt, and zt of each joint data Jn.
Face joints were deemed irrelevant for classifying the exercises and were discarded
 in our collected dataset

### Notes:
The preprocessing part "the encoding" follows as this we store the data in Vector v contating 22 body joints jn each joint contain x,y,z data over time
(we will have to overcome this challenge later as we capture data in 3d)

<!-- Slide number: 15 -->
# Feature Engineering
Feature extraction algorithms are applied to process these joints
5 Statistical techniques are utilized  including :Standard deviation, maximum, median, mean, and minimum
which produced 330 features in total.
FCBF algorithm was employed to rank and select the most significant features.
The model selected the top 20 features and discarded the rest.
It was found that the feature importance score either plateaued or rapidly decreased after the 20th feature.

### Notes:
for feature engineering part we apply- feature extraction using 5 different statistical techniques
- FCBF feature ranking algorithm
- Select only the top 20 feature (why the top 20 ? )

<!-- Slide number: 16 -->
# Expierments
Exp. 1
was conducted to evaluate the performance of the Extra Trees classifier for action recognition on the UI-PRMD dataset, after applying FCBF feature selection.
Exp. 2.
was conducted using our proprietary dataset in a real clinic setting to showcase the system's practicality and effectiveness in real time using only RGB Camera.

<!-- Slide number: 17 -->
# Used Algorithms 1 of 2
Extra Tree
a tree-based ensemble technique used in machine learning.
Extra Trees builds decision trees using the entire dataset.
Trees demonstrate significantly faster performance [6] [7] .
Instead of the greedy approach used in Random Forest, Extra Trees randomly select split values for features.

<!-- Slide number: 18 -->
# Used Algorithms 2 of 2
One Dollar
also known as the 1$ gesture recognition algorithm.
This algorithm converts a gesture into a sequence of points and then calculates the minimum distance between the gesture and a set of predefined templates.
The One Dollar algorithm typically uses two-dimensional (x, y)
However, in this study, the joint coordinates were extracted in 3D.
To overcome this, the X and Y coordinates were summed to create a new X value, while the Z coordinate was assigned to Y. Thus, the data in the tuples were represented as (x+y, z) format.
Due to the limited number of videos in the dataset, the One Dollar algorithm was chosen for this research.

### Notes:
1$ limitation is it accepts tempelate data  in 2d while ours is in 3d format.
hence we overcome this challenge by applying a very famous approach of ccombining  2 coordinates together.

<!-- Slide number: 19 -->
# Expierments 1 of 2
Feature Extraction for 22 body joints.
Ranked and Selected only the top 20 features.
The data was split into 70- 30  for training and testing.
10 iterations.
Applied a cross-validation function with 30 folds.

<!-- Slide number: 20 -->
# Expierments 2 of 2
Feature Extraction for 33 body joints.
Face landmarks were excluded.
Ranked and Selected only the top 20 features.
(x+y, z)
1, 2 , and 3 videos used for training , 4 for testing.

<!-- Slide number: 21 -->
# Results
| Number of Templates | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| One | 72% | 74.5% | 72% | 70.4% |
| Two | 88% | 93% | 88% | 87.9% |
| Three | 90% | 93.8% | 90% | 90.1% |
Evaluation of 1$ algorithm based on different evaluation metrics for different number of templates used
| Algorithm | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| Extra Tree | 99.64% | 99.74% | 99.64% | 99.62% |
| One Dollar | 90% | 93.8% | 90% | 90.1% |
Evaluation of different algorithms based on different evaluation metrics

### Notes:
The first table shows the results of exp2 as 1$ can accept n number of templates , hence we apply different n values for explanation, 1 ,2 and 3 templates .

The three-template combination yielded the ebst results for the dollarpy

The second table shows exp1 results alongside the best results achieved from exp2 . as we notice ET outperformed 1$ in all evaluation metrics . and this also out performed the best recorded accuracy from related work 2

we chose accuracy as the dataset is already balanced

<!-- Slide number: 22 -->
# Conclusion
The performance of the One Dollar algorithm is greatly affected by the number of templates employed for training, as higher numbers of templates for each exercise lead to better outcomes.
Extra Tree Classifier, outperformed the One Dollar algorithm in all four evaluation metrics
Despite having limited training data, the One Dollar algorithm still generated acceptable results.
The Encoding of the data greatly affects the results of the model.
Our Encoding achieved Best recorded results so far.
A limitation of the current pose detection model is its inability to handle scenarios where there are multiple individuals present within the frame

<!-- Slide number: 23 -->
# Future Work
A limitation of the current pose detection model is its inability to handle scenarios where there are multiple individuals present within the frame.
This research has the potential for extension by incorporating wearable sensors in conjunction with the RGB camera to achieve enhanced precision in the results.
The application of various Deep Learning Algorithms with different hyperparameters can be explored to attain improved accuracy.
Employing ensemble learning techniques to combine a set of baseline models can lead to the generation of a more robust and powerful model.

<!-- Slide number: 24 -->
# Reference
W. H. Organization, “Rehabilitation.” Available online, 2023. Accessed on June 15, 2023.
B. Debnath, M. O’Brien, M. Yamaguchi, et al., “A review of computer vision-based approaches for physical rehabilitation and assessment,” Multimedia Systems, vol. 28, no. 2, pp. 209–239, 2022
F. A. Rashid, N. S. Suriani, M. N. Mohd, M. R. Tomari, W. N. W. Zakaria, and A. Nazari, “Deep convolutional network approach in spike train analysis of physiotherapy movements,” in Advances in Electronics Engineering: Proceedings of the ICCEE 2019, Kuala Lumpur, Malaysia, pp. 159–170, Springer Singapore, 2020
N. Tasnim and J.-H. Baek, “Dynamic edge convolutional neural network for skeleton-based human action recognition,” Sensors, vol. 23, no. 2,p. 778, 2023.
A. Vakanski, H.-P. Jun, D. Paul, and R. Baker, “A data set of human body movements for physical rehabilitation exercises,” Data (Basel), vol. 3, Mar. 2018.
M. Fern ́andez-Delgado, E. Cernadas, S. Barro, and D. Amorim, “Do we need hundreds of classifiers to solve real world classification problems?,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3133– 3181, 2014.
R. Caruana and A. Niculescu-Mizil, “An empirical comparison of supervised learning algorithms,” in Proceedings of the 23rd international conference on Machine learning, pp. 161–168, ACM, June 2006.