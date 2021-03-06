# Deep Learning Model Training Time and Performance

## Abstract
This research serves the purpose of understanding deep learning model training time and operational performance across alternative hardware and software platforms. A nested 2x2 experimental design is used to compare the impact on performance using Keras in Python and R, as well as using different central processing units (CPU) in Macintosh operating system (macOS). The experiments are then each employed using one computer vision dataset and one text classification dataset. Overall, the 8-core processor shows much stronger computing power than the dual-core processor, with significant lower processing time in both datasets. On the other hand, R on average processes faster than Python. However, although Python appears to take longer to process, networks employed in Python show high accuracies than networks employed in R. Results indicate that processors have minimal impact on network accuracy.

## Introduction
The objective of this research is to understand the impact of alternative hardware and software platforms on deep learning model training performance. To understand the impacts, models were employed in Python and R, each using a 3.1 GHz Dual-Core Intel Core i5 processor and a 2.3 GHz 8-Core Intel Core i9 processor to observe the differences in performance. Moreover, this research utilized two different types of data, CIFAR-10 computer vision dataset and IMDB movie reviews sentiment classification datasets, to develop different deep neural networks and intends to understand whether the performance is consistent within the platforms. Metrics including processing time, loss, and accuracy were assessed to compare the performance across platforms.

## Codes & Resources

### Datasets
**CIFAR10 small images classification dataset:** https://keras.io/api/datasets/cifar10/  
**IMDB movie review sentiment classification dataset:** https://keras.io/api/datasets/imdb/  
**Python Code Gihub:** https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/codes/python_code.ipynb   
**R Code Gihub:** https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/codes/r_code.R

### Literature Review
The program codes used in this research referenced the examples shown on the Keras official website for Python and R. The deep convolutional neural network structure referenced the ???Train a simple deep CNN on the CIFAR10 small images dataset??? from the Keras Documentation for Python, while the recurrent convolutional network structure referenced the ???Train a recurrent convolutional network on the IMDB sentiment classification task??? example (Chollet, 2015). And to translate the program codes into R, the ???R Interface to Keras documentation??? was used (Allaire, et al.). The documentation provides equivalent Python functions in R language, as well as similar examples.

## EDA
### CIFAR-10 Dataset
**Sample Image:**  
![image 1](https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/program_outputs/cifar10_sample_image.png)

### IMDB Dataset
**Reviews - Class Frequency:**  
![image 2](https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/program_outputs/review_class_frequency.png)
  
**Reviews - Number of Words:**  
![image 3](https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/program_outputs/review_number_of_words_boxplot.png)
![image 4](https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/program_outputs/review_number_of_words_histogram.png)


## Methods
Keras functional API in TensorFlow was utilized to develop deep neural networks and run in Python Jupyter Notebook and RStudio. Both software platforms were employed within the Anaconda environment to limit other possible hardware and software factors that may influence the performance outcomes. To ensure datasets are consistent across software platforms, the preprocessed datasets of CIFAR-10 and IMDB in Keras were utilized, hence minimal data cleaning was needed. Explanatory Data Analysis was performed using Python to understand and visualize the structure of the datasets that will be used in training the networks. The models were also first built in Python to determine the optimal network structures for the experiments. Once the structures of the network have been decided, the exact same models were then fitted using R language.  

A three-way splitting of the data into training, validation (development), and test sets was employed specifically to tune the epoch for early stopping. Early-stopping methodology was implemented to tune the epoch parameters of the networks, where the early stopping criteria was set to monitor the performance of the validation accuracy which the training will stop if the validation accuracy does not improve by 0.01 after 5 epochs.  

A deep convolutional neural network (CNN) was constructed on the CIFAR-10 dataset, with a structure of a block of two convolutional layers followed by a max-pooling layer and a 25% dropout. The block was repeated twice, first with 32 feature maps and then with 64 feature maps. A dense layer with 512 neurons was connected after the convolutional blocks and with another 50% dropout. All hidden layers were trained using rectifier (ReLU) activation function, with the softmax activation function applying to the output layer. On the other hand, a recurrent convolutional network was constructed on the IMDB movie reviews dataset, with the usage of word embedding method. The structure of the network first started with an embedding layer with embedding size of 32 and maximum word length of 250, followed by a 25% dropout. Then a one- dimensional convolutional layer was added, along with a max-pooling layer, followed by a Long Short-Term Memory (LSTM) recurrent layer before the final output layer. All the convoutional layer again was trained using the ReLU activation function, but the sigmoid activation function applied to the output layer in this network.  

Networks in Experiment 1 and 2 were employed within the Dual-core processor, while Networks in Experiment 3 and 4 were employed within the 8-core processor. Networks denoted as A were trained using Python and networks denoted as B were trained using R. Odd number experiments utilized the CIFAR-10 dataset and even number experiments utilized the IMDB dataset. Lastly, all networks were evaluated with performance metrics such as accuracy, loss, and processing time. F1-score (weighted accuracy) was not considered in this research as both datasets are balanced and the F1-score would be very similar to the accuracy.

## Results
Processing time and test accuracy are the main metrics for model assessment. Processing time reflects the computing power of the hardware and efficiency of the software platforms. Overall, the deep CNN on CIFAR-10 dataset requires much longer processing time than the recurrent CNN on IMDB dataset. And as expected, the 8-core processor performs much faster than the dual-core processor, which achieves about 25% faster on the CIFAR-10 dataset and almost 50% faster on the IMDB dataset. Results also show that models are processed faster using R in RStudio than Python in Jupyter Notebook environment, with the exception of CIFAR-10 dataset in 8-core processor where the processing time is almost double for R than Python.  

In terms of network accuracies, the two different processors perform very similarly within same software platform. Networks employed using Python in general show higher accuracies than networks employed in R. However, networks built in Python within the dual-core processor appear to be more overfitted than R within the same processor, while this outcome is reverse within the 8-core processor.  

| Hardware | Software | Dataset | Model | Processing Time (in second) | Train Loss | Train Accuracy | Test Loss | Test Accuracy | 
| :------: |:------: |:------: |:------: |:------: |:------: |:------: |:------: |:------: |
|Dual Core |Python|CIFAR10|1A|2219.9738|0.3676|0.8878|0.711|0.766|
|Dual Core|R|CIFAR10|1B|1737.3885|0.4171|0.8624|0.766|0.7497|
|Dual Core|Python|IMDB|2A|1227.0138|0.1159|0.9726|0.6466|0.8479|
|Dual Core|R|IMDB|2B|562.1390|0.1149|0.9732|0.6551|0.8492|
|8 Core|Python|CIFAR10|3A|981.4050|0.3749|0.8789|0.7066|0.7648|
|8 Core|R|CIFAR10|3B|1891.7623|0.3919|0.8763|0.7384|0.7524|
|8 Core|Python|IMDB|4A|613.1907|0.1133|0.9737|0.6228|0.8568|
|8 Core|R|IMDB|4B|408.6318|0.1355|0.9682|0.7261|0.8407|


![model_results](https://github.com/jessy-chang/deep-learning-model-training-performance/blob/main/program_outputs/model_performance_plots.png)

## Conclusions
Hardware capacity is one of the most important factors when constructing deep learning models, as many of these networks are compute-intensive. When models become more complex, having sufficient hardware computing power will be one of the key factors in the success of constructing a model. However, software platforms are also important in delivering high performing models. Despite using same Keras API, Python appears to achieve higher accuracy than R. In conclusion, the machine with 2.3 GHz 8-Core Intel Core i9 processor shows significant higher computing power than the machine with 2.3 GHz 8-Core Intel Core i9 processor. And R seems to process faster than Python, though Python shows an overall higher accuracy performance.


## License 
I am providing code and resources in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code and resources is from me and not my employer.
```
MIT License

Copyright (c) 2022 Jessy Chang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
