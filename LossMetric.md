# Loss metric

## Hamming loss

Hamming Loss is a metric used to measure the accuracy of binary or multilabel classifications. It calculates the fraction of labels that are incorrectly predicted. The formula for Hamming Loss is given by:

$$ \text{Hamming Loss} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{m} \sum_{j=1}^{m} (y_{\text{true},ij} \neq y_{\text{pred},ij}) $$

where:

- $ N $ is the number of samples.
- $ m $ is the number of labels.
- $ y_{\text{true},ij} $ is the true label of the $ j $-th label of the $ i $-th sample.
- $ y_{\text{pred},ij} $ is the predicted label of the $ j $-th label of the $ i $-th sample.

## Precision

Precision is a metric that measures the accuracy of positive predictions made by a classification model. It is defined as the number of true positives divided by the sum of true positives and false positives. The formula for precision is:

$$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$

## Recall

Recall, also known as Sensitivity or True Positive Rate, is a metric that measures the ability of a classification model to capture all positive instances. It is defined as the number of true positives divided by the sum of true positives and false negatives. The formula for recall is:

$$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$

## Subset accuracy

Subset accuracy, also known as exact match ratio or accuracy_score, is a classification metric that measures the percentage of samples for which the predicted labels exactly match the true labels. In other words, it calculates the accuracy of predicting all labels for a sample correctly. The formula for subset accuracy is:

$$ \text{Subset Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\text{predicted labels}_i = \text{true labels}_i) $$

where:

- $N$ is the number of samples.
- $\mathbb{1}(\cdot)$ is the indicator function that returns 1 if the condition inside is true and 0 otherwise.

## Negative predictive value

Negative Predictive Value (NPV) is a metric that assesses the ability of a classification model to correctly identify negative instances. It represents the proportion of true negatives among instances predicted as negatives. The formula for Negative Predictive Value is:

$$ \text{NPV} = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Negatives}} $$

Here's the Python implementation of Negative Predictive Value:

## F-measure

### F1 score

F-measure, also known as F1 score, is a metric that combines precision and recall into a single value. It is particularly useful when there is an uneven class distribution (class imbalance) in a classification problem. The F1 score is the harmonic mean of precision and recall and is defined as:

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

where:

- Precision is the number of true positives divided by the sum of true positives and false positives.
- Recall is the number of true positives divided by the sum of true positives and false negatives.

The F1 score ranges from 0 to 1, with higher values indicating better performance.

### F-beta score

F-beta score is a generalization of the F1 score that introduces a parameter $\beta$ to control the emphasis on precision or recall. The formula for F-beta score is:

$$ F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}} $$

Here, $\beta$ is a positive constant. If $\beta > 1$, the F-beta score will favor recall over precision, and if $\beta < 1$, it will favor precision over recall. When $\beta = 1$, it is equivalent to the F1 score.

## Markesness

## Informedness
