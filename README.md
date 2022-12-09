# Got Headlines: Abstractive Title generation for News articles
## Usage
Run the experiments by doing:\
`cd src` first and then running one of:\
`python3 main.py [experiment name]` to run the training and evaluation for the generalization experiments.\
`python3 practical_year_model.py --train-year [year number]` to run the training and evaluation experiments for the year-by-year experiments. Requires the model from the previous year to be trained first for any given year.\
`python3 evaluate_model.py [experiment name]` to run the evaluation experiments for the generalization experiments on all the out of distribution articles using the model trained for general articles.\\

## Bash Script Usage
If on an HPC, the above can be run by submitting one of the following jobs:\
`deepspeed_startup.sh` to run the generalization experiments using DeepSpeed (requires the data to be tokenized first without deepspeed) \
`startup.sh` to run the generalization experiments without parallelizattion\
`fewshot_startup.sh` to run the out of distribution evaluations for generalization experiments \
`year_train.sh` to run the year-by-year training experiments \
## Experiments names possible

`generic_allyears`: Extract articles only from the publications CNN, Reuters and The New York Times, for all years and train/evaluate on T5\
`peg_ally`: Extract articles only from the publications CNN, Reuters and The New York Times, for all years and train/evaluate using Pegasus\
`pegx_ally`: Extract articles only from the publications CNN, Reuters and The New York Times, for all years and train/evaluate using Pegasus-X\
`pegx_ally2019`: Extract articles only from the publications CNN, Reuters and The New York Times, for all years and train/evaluate using Pegasus-X

