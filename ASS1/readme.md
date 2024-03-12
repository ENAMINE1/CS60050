# README

This README provides an overview of the Naive Bayes Classifier implemented in Python for the given dataset.

### Group Number
- 42

### Roll Numbers
- 21ME30078 Debraj Das
- 21CS30047 Shashwat Kumar

### Project Number
- NANB

### Project Title
- Nursery School Application Selection using Naive Bayes Algorithm

## Setup

To run the code successfully, follow these steps:

1. Clone the repository or download the `nursery.csv` dataset.
2. Ensure the dataset file (`nursery.csv`) is in the same directory as the Python script.
3. Install the required dependencies using pip by running:
    ```
    pip install -r requirements.txt
    ```
4. Execute the Python script (`naive_bayes_classifier.py`).
    ```
    python naive_bayes_classifier.py
    ```

## Requirements

You can find the list of required dependencies in the `requirements.txt` file. Install them using the command mentioned above.

## How to Run the Code

The script will load the dataset, preprocess it, train the custom Naive Bayes Classifier, and compare its performance with Scikit-learn's Naive Bayes Classifier.

## Methodology

1. **Data Loading and Preprocessing**:
   - The dataset (`nursery.csv`) is loaded using Pandas.
   - Categorical variables are converted to numerical using label encoding.

2. **Custom Naive Bayes Classifier**:
   - A custom Naive Bayes Classifier is implemented from scratch.
   - It calculates class probabilities and feature probabilities using Laplace smoothing.

3. **Scikit-learn's Naive Bayes Classifier**:
   - The dataset is split into training and testing sets using Scikit-learn's `train_test_split`.
   - Scikit-learn's Gaussian Naive Bayes Classifier is trained and tested.

4. **Performance Comparison**:
   - The accuracy and classification report are printed for both custom and Scikit-learn's Naive Bayes Classifiers.

## Insights

- The code demonstrates the implementation of a Naive Bayes Classifier from scratch and compares its performance with Scikit-learn's implementation.
- It provides insights into the effectiveness of the custom implementation compared to the library-provided solution.

## Additional Notes

- Both classifiers are trained on the same dataset and evaluated using the same test set for fair comparison.
- Experimentation with hyperparameter tuning can further optimize the performance of both classifiers.

## Support

For any questions or issues, please contact 
- [Shashwat Kumar](kshashwat.iit@gmail.com).
- [Debraj Das](susmita834805@gmail.com)