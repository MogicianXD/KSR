### Difference

For our paper, we adjust some of the pattern.

- Instead of **mini-batch** which GRU4rec and the original KSR use, We adopt '**data augmentation**', the same as SR-GNN, STAMP, etc. (use [0:k] for k in range(1, len(session)) for train, and [k] for target).
- On our dataset, we found that **NLL** loss performs better than KSR with BPR loss, contrary to the original. Here, we provide **NLL** codes with **BPR** codes annotated.
- You need to adjust the **dataset path**. Ours is "../benchmarks/kg_month15/A/"

### Attention

- Without pretraining, you can randomly generate embbeding based on normal distribution (default for item emb). Kg emb pretrained with trans-E, we get a faster loss convergence with metrics a little bit higher . We still use the original code to read local embedding files,  you should also generate three embedding files, of which the format is not different from [the original](https://github.com/RUCDM/KSR/blob/master/data/the format of KSR input.txt) .
- To speed up, we begin to valid when epoch > 20; model is saved when it obtains the highest MRR@20. You can alter this in the method "fit".
- As the original version filters out items in test data which not exist in train data (the same way as GRU4rec), we follow it. If you want metrics be divided by the sum of unfiltered test sessions (passed as an arg), you can revise it in evaluation.py.

### Input Format

##### sess data:

The first line is "SessionId,ItemId". Id begins with 0.

> SID+','+IID

##### item_id -> kg_id

ItemId is mapped to entityId in kg for consistency. No header. 

> IID+'\t'+EID

### Args

optional arguments:

-h, --help            show this help message and exit

--data DATA           dataset folder name under ../benchmarks

--type TYPE           our dataset is divided

--epochs EPOCHS       Number of epochs.

--batch_size BATCH_SIZE	Batch size.

--dropout DROPOUT     Dropout rate.

--out_dim OUT_DIM     Embedding size of output.

--num_neg NUM_NEG     Number of negative instances to pair with a positive instance.

--lr LR               Learning rate.

--activation [ACTIVATION] Specify activation function: sigmoid, relu, tanh, identity

--momentum MOMENTUM   Momentum as hyperprameter.

--argument            use the method called "data argument" if true

--pretrain             use kg emb, the pretrained output of OpenKE (trans-E)

--reload              restore saved params if true

--eval                only eval once, non-train

--save                if save model or not

--savepath SAVEPATH   for customization

--cuda CUDA           gpu No.





