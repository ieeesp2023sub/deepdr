# DeepDR
## Introduction
![Workflow of DeepDR](https://github.com/ieeesp2023sub/deepdr/blob/main/deepdr_overview.PNG)
DeePDR aims at detecting fine-grained attack activities and linking them as the attack campaign. This figure shows the architecture of DEEPDR, which consists of four phases: (a) audit log pre-processing; (b) event representation learning; (c) attack event detection; and (d) attack campaign recovery. These phases include five components: (1) Log Parser (LP), (2) Event Correlation Network (ECN), (3) Event Representation Network (ERN), (4) Attack Event Detector (AED) and (5) Attack Campaign Recovery (ACR). <br />
**Audit Log Pre-processing.** LP first standardizes the format of various types of log events, uniformly vectorizes these standardized events, and arranges these processed events as a sequence in chronological order. Then, for the following event representation learning and attack event detection, LP extracts relevant event pairs, irrelevant event pairs and contextual windows of each event by moving a sliding window of a specific size on the sequence.<br />
**Event Representation Learning.** First, ECN inputs the extracted event pairs by LP, and learns correlations (i.e., attackrelevant correlation, benign-relevant correlation and irrelevant correlation) of event pairs based on events’ labels. Then, ERN builds relevant window (i.e., attack-relevant window and benign-relevant window) for each event through the recognized correlations by ECN. Next, ERN learns each event’s representation based on the built relevant window and events’ labels. Moreover, AED learns the representation distributions of attack events and benign events respectively.<br />
**Attack Event Detection.** Given an unlabeled event and its contextual window, ECN first recognizes the correlations between the event and its contextual events, and then builds the attack-relevant window and benign-relevant window based on the correlations. Then, ERN represents the event both as a potential attack event and as a potential benign event using the event’s attack-relevant window and benign-relevant window respectively. Finally, AED determines the event’s label according to the compliance of the event’s representations with the previously learned representation distributions.<br />
**Attack Campaign Recovery.** ACR recovers an attack campaign based on the correlations of the detected attack events given by ECN. Those detected events that cannot be correlated to any other events are filtered as false positives.

## Requirements
Python Version：2.7  

Dependent packages:
> tensorflow==1.10.0
> networkx==2.2  
> numpy==1.15.1  
> scipy==1.0  

## Usage
### Input
1. Log file path.
2. POI.
### Command
Execute the following command from the project home directory ``DEPCOMM``:<br/>

	./core/Start "log-file-path" "poi" 

For example, to run depcomm on the unzipped example log file, first put the unzipped example log file into the folder ``input`` in the project home directory ``DEPCOMM``, then execute the following command:

    ./core/Start "./input/leak_data.txt" "10.10.103.10:38772->159.226.251.11:25"

### Data

Due to the limit of Github, we can't upload the collected extreme large log files.
The folder example contains a small log that can be used for demo.
For this case, the POI event is a suspicious network connection.
For the DARPA Attack used in evaluation, here is the [github link](https://github.com/darpa-i2o/Transparent-Computing). 
You can follow their instructions to download data.  
### Output
DepComm will output several different files to the folder ``output`` in the project home directory ``DEPCOMM``.
1. Some dot files that are named as ``community_*.dot``. They are the graphs for each community. 
2. A dot file that is named as ``summary_graph.dot``. It is the summary graph, where the node denotes community and the edge denotes the data flow direction among communities.
3. A txt file that is named as ``summary.txt``. It contains the master process, time span and prioritized InfoPaths for each community.
4. Another txt file that is named as ``community.txt``. It records nodes attributes for each community (pid and pidname for process node, path and filename for file node, IP and Port for network node).  

Note that the dot files can be visualized by [Graphviz](https://github.com/xflr6/graphviz).
