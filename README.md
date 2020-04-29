#### Project Type : Minor Project
#### Project Title : Leveraging Software Defined Networks to mitigate      DDoS Attacks

####Team Members:
+ Abhilash Jaysheel (01FB16ECS009)
+ Abhimanyu Roy (01FB16ECS010)
+ Aditya Girisha (01FB16ECS027)

####Project Guide: Prof Suganthi

##Project Abstract:

Software-defined networking (SDN) is a new network architecture that has been proven to enhance network performance and reliability. SDN networks promote logically centralized control of network switches and ro
uters in an SDN environment by separating network behaviour and network functionality.

The Aim our project is to leverage this capability by detecting signatures that indicate traces of a DDoS attack and dynamically set flow rules that prevent forwarding of these packets.By dropping these black packets we can keep server utility to a minimum even during the event of a DDoS attack making it available for processing genuine user requests.

###Code Execution

####ML Classifier
The ML Classifier is a numpy-based python prototype of an feed-forward artificial neural network developed for integration with the RyuSDn network.

The feedforward network is accomanied by a back-propagation for efficient learning and k-fold cross-validation for confirming accuracy.

Activation Function:
$$ S(x) = {\frac {1}{1+e^{-x}}} $$

Activation Derivative:
$$ Sd(x) = {\frac {1}{1+e^{-x}}}* (1- {\frac {1}{1+e^{-x}}}) $$



####Requirements
+ python3
+ conda
+ numpy
+ pandas
+ pickle
+ sklearn

######Install python3:
` $ sudo apt-get install python3.6`
######Install conda:
`$ wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh`

`$ bash Anaconda3-2020.02-Linux-x86.sh`
######Install python requirements
`$  pip install -r requirements.txt`

####Execution
+ The classifier  implementation can be found under `ML/src/NN.py`
+ The training of the classfier can found under `ML/main.ipynb`
+ The trained classifier can be tested under `ML/demo.ipynb`
+ The trained network is stored under `ML/model/save.pkl` which is required to execute the trained classifier. if absent execute main.ipynb

To run the the demo or training code run :
`$ jupyter notebook` and open the files respectively to execute.
 ######or
`$ jupyter nbconvert --to python *.ipynb`
`$ python3 *.py`
where `*.ipynb` is the file to be executed.
