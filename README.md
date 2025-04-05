# CS4413_Group_Project_AI

## __Contributers:__

_Matthew Chiasson_,
_Joel David English_,
_Michael Gilmore_,
_Matias Gomez Salvarredi_

## Imact of DP and HE on Model preformance and effitency.

# Purpose

The proliferation of AI and deep learning models has brought with
it an ever-increasing need for data privacy protections. Federated
Learning (FL) is one such model and its real-world potential and
capacity for decentralized learning has fueled its current popularity.
Recently, research has exposed vulnerabilities in FL which make
establishing proper privacy protections for the model a priority. To
satisfy that need our objective was to examine the application of
differential privacy (DP) in various ways to provide a layered privacy 
defense to an FL system,
against a variety of attack vectors, including model inversion and
data poisoning. 

Implementation of these goals proved more
challenging than expected. challenges with DP
implementation prevented proper testing across all but a few target
study areas. Findings suggest further work or alternative protection
schemes would be needed to provide adequate privacy protection
against both model inversion attacks and data poisoning. Future
work would also include resolution of the incompatibilities
discovered during implementation.

this GitHub repo Contains the code for our implementation of the Fedarated learning client, DP implementation, Data poisioning attacks, and Model inversion attacks for this project.

# Method

Our approach was to develop a federated learning model to classify
real-world images. This model will then be subjected to various
attack vectors. For each attack vector, the model would utilize some
combination of DP test and compare its protective capabilities. Defensive schemes would also
be explored for different areas of the model, such as protecting the
dataset or model updates, or both. Initial focus would be on model
inversion and gradient inversion protection, with later testing on
membership inference.
Our objective is both to establish privacy protection capability, and
to measure the trade-offs present between privacy protection and
model performance


# Implementation

The federated learning model was constructed using pretrained
ResNet18 on the PyTorch framework. Flower library was used for
instantiation and management of FL clients; Pandas libraries
provided for data manipulation. Differential privacy was
implemented using the PySyft library. Data poisoning attacks were
implemented without the use of special libraries; model inversion
attacks (MIA) were setup using a utility function which called the
trained model (both with and without DP applied) and executed an
attack on it. 

See the Train_ResNet18_FL.py script for training this model.

Datasets were split evenly across the number
of instantiated FL clients. Differential privacy was applied to local
model gradients only, while poisoning attacks consisted of label
flipping on a single client for a single pair of test classes (bird
images). Model inversion attacks initialized a random noise
‘image’ and performed prediction iterations a preset number of
times on either a DP protected trained model or an unprotected
trained model. Models were loaded into the MIA script with state
dictionary key mismatch correction to ensure Pytorch
compatibility. Regularization techniques (L2 loss and total
variation loss) were used to help produce smoother, more realistic
output images.

Number of clients and training rounds varied depending on the test,
but for DP and poisoning testing consisted of five clients training
over ten rounds. Model inversion testing was simulated on a single
client model. Differential privacy was activated as needed
depending on the test. Data poisoning was only activated for 
poisoning tests and was applied to two of the five clients.




# How to Use

## Train_Resnet18_FL.py

**Authors:**
 
_Matthew Chiasson:_ FL Network implementation,

_Joel David English:_ DP, and data poisoning implementation


Run the mian program 'Train_ResNet18_FL' to run a Fedarated Learning simulation that uses the pretrained Resnet18 model wieghts. You will need to use command line arguments to configure how the model will train.

the command line arguments are detailed as follows:

> --csv_path: type str, required, Description: Path to your dataframe.csv.

> --UseDP: type=str, defaults to "false" (Not requierd), Description: Weather to use DP in training or not.

> --poison_active: type: str, defaults to: "false" (Not requierd), Description: Weather to use Data poisoning in training or not.

> --num_rounds: type: int, defaults to 5 (Not requierd), Description: Number of federated learning rounds.

> --num_clients: type: int, defaults to: 5 (Not reqired), Description: Number of simulated clients.

> --batch_size:  type: int, defaults to: 32 (Not reqired), Description: Batch size for clients.

> --save_path: type: str, defaults to "Train_Resnet18_FL.pth", Description: Path to save the updated model.

An example usage is:

python Train_ResNet18_FL_DP.py --csv_path "cub17_dataframe.csv" --UseDP true --poison_active true --num_rounds 10 --num_clients 5 --batch_size 32 --save_path "ResNet18_Victum.pth"

## MIA.py (Model Invresion Attack)

__Author: Michael Gilmore__

No command line argumants. You will be prompted to chose from 1 of 2 victum models. model1, trained model without Differintial privacy. Or model2, trained model with DP. You will need to have the selected model.pth file saved with the appropriate name in the MIA directory.

You will then be prompted for the class you would like to poison.

Wait and recive results. Results will include a random example image taken from the class you chose on the left, and the MIA genarated image on the right.

## Homomorphic encryption.py

**Authors:**

__Matthew Chiasson__: Fedarated learning Framework skeleton, Modified Resnet18.
__Michael Gilmore__: HE implementation

Uses command line aguments:
the command line arguments are detailed as follows:

> --model_path: type: str, required, Description: Path to the existing model file (.pth).

HE uses a Modified Resnet18 architecture trained in ModifiedTrainer.py. This modified trainer Includes A dropout layer and a few other changes we wanted to experament with.

> --csv_path: type str, required, Description: Path to your dataframe.csv.

> --num_rounds: type: int, defaults to 5 (Not requierd), Description: Number of federated learning rounds.

> --num_clients: type: int, defaults to: 5 (Not reqired), Description: Number of simulated clients.

> --batch_size:  type: int, defaults to: 32 (Not reqired), Description: Batch size for clients.

> --save_path: type: str, defaults to "Train_Resnet18_FL.pth", Description: Path to save the updated model.


# Dataset

Dataset CUB-200-2011 can be fond [here](https://www.vision.caltech.edu/datasets/cub_200_2011/)

The dataset, CUB-200, later
modified to CUB-17 was chosen due to high usability score and
built in bounding box, which should have improved HE
application. The drawback of this dataset was limited images per
class in the dataset. This was remedied by adding transformations
to each image during training to artificially expand the training set by adding 
small transformations in color jitter, image rotation, and.
Initial training on the original dataset revealed noticeable feature
inconsistency, causing the model to consistently perform worse
than expected. This was resolved by trimming classes with the
lowest training generalization. The dataset training subset was also
smaller than anticipated. While both issues would be resolved by
the final version of the model, testing might have gone differently
with a larger, more consistent dataset.

# Results

To test the implementation of DP and Data poisoning we ran Train_ResNet18_FL.py with the following paramiters:

- 3 models were trained without DP and wothout Data Poisoning to observe a controll group.
- 3 models were trained with Dp and without Daata Poisoning to observe the impact of DP on the models accuracy.
- 3 models were traind without DP and with Data Poisoning to observe Data poisonings impact on a single labe.
- 3 Models where trained with both DP and Data Poisoning to observe the Protection that DP provieds against data poisoning.
>
Each Model was trained with 10 rounds and 5 clients. In the cases where Data poisoning is active

The resulting models from theas tests can be found in the form of multiple classification repots; Saved under the 'Results' folder. The trained models for thease results can be found in the 'Results Models' folder.

Each classification report is a collection of 3 Json objects; Each object represents a model that was trained with the respective reports specified training paramiters I.E. used DP, Used data poisioning. Each object also details the persision, recall, f1-score, and support, for each class in the dataset. Additionaly, each report holds more relavent data on how that model as trained such as number of training rounds, number of clients, DP nise multiplyer, etc.

Example:

        "15": {
            "precision": 0.9655172413793104,
            "recall": 0.9655172413793104,
            "f1-score": 0.9655172413793104,
            "support": 29.0
        },
        "16": {
            "precision": 0.8529411764705882,
            "recall": 1.0,
            "f1-score": 0.9206349206349206,
            "support": 29.0
        },
        "accuracy": 0.9369158878504673,
        "macro avg": {
            "precision": 0.9452701393997567,
            "recall": 0.9357301401926148,
            "f1-score": 0.9386188811444746,
            "support": 428.0
        },
        "weighted avg": {
            "precision": 0.9407660459912061,
            "recall": 0.9369158878504673,
            "f1-score": 0.9369846496905702,
            "support": 428.0
        },
        "Model Name": "NM_ResNet18_Victum_NDP_NP_1",
        "Use DP": "false",
        "noise multiplier ": "NAN",
        "max grand norm": "NAN",
        "poison active": "false",
        "num rounds": 10,
        "num clients": 5,
        "num poisoned clients": 0,
        "poisoned label": "NAN"

The results from the 3 runs where avrages and put into a tale to view the affects of DP and data poisoning:

Classes 6 and 10 where shown and the rest where ommited because all instances of label 6 was replaced with label 10 in the poisoned clients.

| Condition          | Trial 1 | Trial 2 | Trial 3 | Trial 4 |
| ------------------ | ------- | ------- | ------- | ------- |
| DP enabled         | No      | Yes     | No      | Yes     |
| Poisoning          | No      | No      | Yes     | Yes     |
| Class 6 Precision  | 0.946   | 0.688   | 0.986   | 0.909   |
| Class 10 Precision | 0.967   | 0.883   | 0.915   | 0.801   |
| Global Precision   | 0.927   | 0.812   | 0.913   | 0.789   |
| Class 6 Recall     | 0.956   | 0.789   | 0.656   | 0.256   |
| Class 10 Recall    | 0.978   | 0.978   | 0.967   | 0.933   |
| Global Recall      | 0.921   | 0.803   | 0.900   | 0.775   |
| Class 6 f1-score   | 0.950   | 0.735   | 0.770   | 0.396   |
| Class 10 f1-score  | 0.972   | 0.927   | 0.937   | 0.844   |
| Global f1-score    | 0.921   | 0.797   | 0.898   | 0.761   |
| Global accuracy    | 0.921   | 0.804   | 0.900   | 0.775   |

Expectations before the trials was class 6 precision would stay
relatively stable but decline when DP was applied, with recall
declining due to the poisoning attack. Class 10 was expected to
decline in precision, and to a lesser degree in recall across all trials.
Results show overall performance decline with DP application over
the base model, and a minor decline over the base model when the
dataset is poisoned. We note the recall performance of class 6,
specifically for Trial 4 is not as expected, with a dramatic decline
from any other trial. Another notable result is class 10 recall stays
very stable throughout all trials, which is contrary to expectation.