### Difference

For our paper, we adjust some of the pattern.

- Instead of **mini-batch** which GRU4rec and the original KSR uses, We adopts "**data augmentation**", the same as SR-GNN, STAMP, etc. (use [0:k] for k in range(1, len(session)) for train, and [k] for target).
- On our dataset, we found that **NLL** loss performs better than KSR with BPR loss, contrary to the original. Here, we provide **BPR** codes with NLL codes annotated.
- You need to adjust the **dataset path**. Ours is "../benchmarks/kg_month15/A/"

### Attention

- It's not necessary to pretrain embbeding (emb can be updated when backwarding, different from the original code) . Thus, we **randomly generate embbeding** based on normal distribution (default for item emb). Kg emb pretrained with trans-E, we get a faster loss convergence with metrics higher a bit. However, we still use the original code to read local embedding files,  you should also generate three embedding files, of which the format is not different from [the original]( https://github.com/BetsyHJ/KSR/tree/master/data ).
- To speed up, we only train when epoch <= 40; model is saved when epoch > 50 and it obtains the highest MRR@20. You can alter this in the method "fit".
- As the original version filters out items in test data which not exist in train data (the same way as GRU4rec), we follow it. But metrics is divided by the sum of  unfiltered test sessions. You can revise it in evaluation.py.

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

--retrain             use kg emb, the pretrained output of OpenKE (trans-E)

--reload              restore saved params if true

--eval                only eval once, non-train

--save                if save model or not

--savepath SAVEPATH   for customization

--cuda CUDA           gpu No.





