Imagine a table where the rows are the subjects and the features are the columns. 
Features are in order of most important to least important. 
Subjects are in order of youngest to oldest. 
Now for a single feature, there will be column with some minimum and some maximum. This is normalized on a scale from 0-1 on a color bar. 
So given a specific brain sample, it will plot the first n centers according to their feature importance, and then color them individually according to the 0-1 scale.

        most important ----> least important
......... Subject#  Feature1   Feature2   Feature3 ...
 youngest     1       val1       val2       val3
    |         2       val4       val5       val6
    |         3       val7       val8       val9
    |         4       val10      val11      val12
    V         .
 oldest       .
              .


viz_age_weights.py generate a slider animation for weights
viz_age_covariances.py generate a slider animation for weights
gif[ ].py files generate the gifs for the same sliders where weight is fixed at 250 Gaussians and covariances is fixed at 10 Gaussians
